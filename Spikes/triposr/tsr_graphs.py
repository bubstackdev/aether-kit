"""TripoSR's graphs as the exporter, the eager references and the equivalence check build them. `m` is a TSR model (for
the field queries, anything with its `decoder`, `renderer.cfg.radius` and `renderer.cfg.density_bias` will do).

  Scene      image [1, 3, 512, 512] → scene codes [3, 40, 64, 64], with the 32×32 position embedding precomputed
  Field      scene codes + points [N, 3] → density, colour, sampling the triplanes with an explicit gather
  GridField  scene codes + grid coordinates → density over a z-slab, sampling with hat-function matmuls (no gather)

Core AI can't lower grid_sample, which upstream's renderer uses; both field queries reproduce it exactly
(check_triplane_sampling.py).
"""
import torch
import torch.nn.functional as F


class Scene(torch.nn.Module):
    """Image → scene codes with the 32×32 interpolated position embedding precomputed (fixed 512 input)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        vit = m.image_tokenizer.model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            pos = vit.embeddings.interpolate_pos_encoding(vit.embeddings.position_embeddings.new_zeros(1, 1025, 768),
                                                          512, 512)
        self.register_buffer("pos", pos.clone())
        vit.embeddings.interpolate_pos_encoding = lambda embeddings, h, w: self.pos.to(embeddings.dtype)

    def forward(self, image):
        tok = self.m.image_tokenizer
        x = (image[:, None] - tok.image_mean.to(image.dtype)) / tok.image_std.to(image.dtype)
        o = tok.model(x[:, 0], interpolate_pos_encoding=True)
        img_tokens = o.last_hidden_state                                   # [1, 1025, 768]
        tokens = self.m.tokenizer(1)                                        # [1, 1024, 3072]
        tokens = self.m.backbone(tokens, encoder_hidden_states=img_tokens)
        return self.m.post_processor(self.m.tokenizer.detokenize(tokens))[0]   # [3, 40, 64, 64]


class Field(torch.nn.Module):
    """Triplane density/color query with explicit bilinear sampling (grid_sample, align_corners=False, zero pad)."""

    def __init__(self, m):
        super().__init__()
        self.decoder = m.decoder
        self.radius = float(m.renderer.cfg.radius)
        self.density_bias = float(m.renderer.cfg.density_bias)

    def forward(self, planes, points):
        P, C, H, W = planes.shape                                   # 3, 40, 64, 64
        padded = F.pad(planes, (1, 1, 1, 1))                         # zero border = grid_sample zeros padding
        Hp, Wp = H + 2, W + 2
        flat = padded.permute(0, 2, 3, 1).reshape(P * Hp * Wp, C)    # [3·66·66, 40]
        x = points / self.radius                                     # (-1, 1)
        feats = []
        for p, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):        # grid (x → W, y → H) = (coord a, coord b)
            u = ((x[:, a] + 1) * W - 1) * 0.5 + 1                    # padded pixel coordinates
            v = ((x[:, b] + 1) * H - 1) * 0.5 + 1
            u0, v0 = torch.floor(u), torch.floor(v)
            wu, wv = (u - u0)[:, None], (v - v0)[:, None]
            u0i, v0i = u0.to(torch.int32), v0.to(torch.int32)
            base = p * Hp * Wp
            def g(vv, uu):
                return flat.index_select(0, (base + vv * Wp + uu).reshape(-1))
            f = (g(v0i, u0i) * (1 - wu) * (1 - wv) + g(v0i, u0i + 1) * wu * (1 - wv)
                 + g(v0i + 1, u0i) * (1 - wu) * wv + g(v0i + 1, u0i + 1) * wu * wv)
            feats.append(f)
        h = torch.cat(feats, dim=-1)                                 # [N, 120], plane-major like "Np Cp -> (Np Cp)"
        o = self.decoder.layers(h)
        density = torch.exp(o[:, 0:1] + self.density_bias)
        color = torch.sigmoid(o[:, 1:4])
        return density, color


class GridField(torch.nn.Module):
    """Density over a z-slab of the regular grid TSR.extract_mesh builds, without a gather. A bilinear sample with
    align_corners=False and zero padding at padded-texel coordinate u is sum_p max(0, 1 - |u - p|) · texel[p], so each
    plane's features over a grid are W_b · plane · W_aᵀ with hat-function weight matrices W. Plane-major features and
    the decoder match Field exactly."""

    def __init__(self, m):
        super().__init__()
        self.decoder = m.decoder
        self.radius = float(m.renderer.cfg.radius)
        self.density_bias = float(m.renderer.cfg.density_bias)

    def weights(self, coords, size):                                  # coords [N] in (-r, r) → [N, size + 2]
        u = ((coords / self.radius + 1) * size - 1) * 0.5 + 1         # padded pixel coordinate, as in Field
        texel = torch.arange(size + 2, dtype=coords.dtype)
        return torch.relu(1 - (u[:, None] - texel[None, :]).abs())

    def forward(self, planes, xs, zs):
        P, C, H, W = planes.shape
        padded = F.pad(planes, (1, 1, 1, 1))                          # [3, C, H + 2, W + 2]
        # x and y share the grid coordinates; each axis is weighted along the plane dimension it indexes.
        x_w, y_h, y_w, z_h = self.weights(xs, W), self.weights(xs, H), self.weights(xs, W), self.weights(zs, H)

        def sample(plane, rows, cols):                                  # [C, H+2, W+2] → [len(cols), len(rows), C]
            # Two linear layers with the hat weights as the weight matrix. coreai-torch 0.4.2 miscompiles the
            # equivalent einsum ("jv,cvu,iu->ijc": wrong values, no error), and a broadcast matmul or an expanded bmm
            # fails its optimizer (diag_grid.py).
            t = F.linear(plane.transpose(1, 2), rows)                   # [C, W+2, Nrows]
            t = F.linear(t.transpose(1, 2), cols)                       # [C, Nrows, Ncols]
            return t.permute(2, 1, 0)

        f0 = sample(padded[0], y_h, x_w)                                # plane (x → W, y → H): [R, R, C]
        f1 = sample(padded[1], z_h, x_w)                                # plane (x → W, z → H): [R, S, C]
        f2 = sample(padded[2], z_h, y_w)                                # plane (y → W, z → H): [R, S, C]
        R, S = xs.shape[0], zs.shape[0]
        h = torch.cat([f0[:, :, None, :].expand(R, R, S, C), f1[:, None, :, :].expand(R, R, S, C),
                       f2[None, :, :, :].expand(R, R, S, C)], dim=-1)   # [R, R, S, 3C]
        o = self.decoder.layers(h.reshape(R * R * S, 3 * C))
        return torch.exp(o[:, 0] + self.density_bias).reshape(R, R, S)
