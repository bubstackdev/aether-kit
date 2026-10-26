"""Export-friendly Kokoro-82M graphs and the shared reference harness.

Kokoro (hexgrad/Kokoro-82M, Apache-2.0) is a StyleTTS2/iSTFTNet model. Its upstream
PyTorch code (kokoro 0.9.4, Apache-2.0) is imported read-only from an unpacked wheel;
nothing here modifies it. Upstream's package __init__ imports the misaki G2P pipeline,
so this harness registers an empty ``kokoro`` package and loads only the model modules.
No grapheme-to-phoneme code (and therefore no espeak-ng) is imported.

The export splits the upstream ``KModel.forward_with_tokens`` at its data-dependent point:

* ``durations``: token ids [1,T] + style [1,256] -> per-token duration sums [1,T]
  (before speed division and rounding), prosody features d [1,T,640] and text
  features t_en [1,512,T].
* host: round(duration / speed) half-to-even, clamp >= 1, and gather the frame
  alignment (upstream multiplies by a one-hot matrix; a gather is exact).
* ``decoder``: aligned prosody en [1,640,F], aligned text asr [1,512,F], style
  [1,256] and standard-normal source noise [1,F,600,9] -> audio [1,600F] at 24 kHz.

Primitives Core AI does not lower are replaced by numerically equivalent forms:

* LSTMs run as a ``while_loop`` recurrence (both directions in one batch-2 loop).
  Upstream packs/unpacks sequences; for batch 1 without padding that is the identity.
* Nearest/linear 1-D upsampling uses reshape/expand arithmetic that reproduces
  PyTorch's ``align_corners=False`` sampling for integer factors.
* ``torch.stft``/``torch.istft`` (complex) become exact real convolutions: reflect
  padding, one-sided DFT, irfft weighting (DC/Nyquist once, other bins twice) and
  division by the overlap-added squared-window envelope. Upstream's own
  ``CustomSTFT`` is an approximation and is not used.
* SineGen's random draws become an explicit input. Its random initial phase has no
  effect upstream: the following linear downsampling by 300 never samples index 0.
"""
from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 600  # decoder upsamples F0 x2, generator x(10*6), iSTFT hop 5
HARMONICS = 9
N_FFT = 20
HOP = 5
MAX_PHONEMES = 510  # 512 ALBERT positions minus the two boundary tokens


def load_upstream(upstream_root: Path):
    """Imports kokoro.model/istftnet/modules without running kokoro/__init__.py."""
    expected = str(upstream_root / "kokoro")
    if "kokoro" in sys.modules:
        existing = sys.modules["kokoro"]
        # Refuse a real kokoro install: its package __init__ imports the misaki G2P pipeline.
        if getattr(existing, "__file__", None) is not None or list(getattr(existing, "__path__", [])) != [expected]:
            raise RuntimeError("a different 'kokoro' package is already imported; run the harness in a fresh interpreter")
        return sys.modules["kokoro.model"], sys.modules["kokoro.istftnet"], sys.modules["kokoro.modules"]
    if "loguru" not in sys.modules:  # upstream only uses logger.debug; avoid an extra install
        stub = types.ModuleType("loguru")

        class _Logger:
            def __getattr__(self, name):
                return lambda *a, **k: None

        stub.logger = _Logger()
        sys.modules["loguru"] = stub
    package = types.ModuleType("kokoro")
    package.__path__ = [expected]
    sys.modules["kokoro"] = package
    model = importlib.import_module("kokoro.model")
    istftnet = importlib.import_module("kokoro.istftnet")
    modules = importlib.import_module("kokoro.modules")
    return model, istftnet, modules


def build_kmodel(upstream_root: Path, snapshot: Path):
    model, _, _ = load_upstream(upstream_root)
    kmodel = model.KModel(repo_id="hexgrad/Kokoro-82M", config=str(snapshot / "config.json"),
                          model=str(snapshot / "kokoro-v1_0.pth"), disable_complex=False)
    return kmodel.eval()


# ---------------------------------------------------------------------------
# Primitives


class LoopLSTM(nn.Module):
    """Bidirectional single-layer batch-first LSTM for batch 1, as a while_loop."""

    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        assert lstm.bidirectional and lstm.num_layers == 1 and lstm.batch_first and lstm.bias
        h = lstm.hidden_size
        self.hidden = h
        w_ih = torch.stack([lstm.weight_ih_l0, lstm.weight_ih_l0_reverse]).detach()  # [2,4H,I]
        w_hh = torch.stack([lstm.weight_hh_l0, lstm.weight_hh_l0_reverse]).detach()  # [2,4H,H]
        b_ih = torch.stack([lstm.bias_ih_l0, lstm.bias_ih_l0_reverse]).detach()
        b_hh = torch.stack([lstm.bias_hh_l0, lstm.bias_hh_l0_reverse]).detach()
        self.register_buffer("w_ih_t", w_ih.transpose(1, 2).contiguous())  # [2,I,4H]
        self.register_buffer("w_hh_t", w_hh.transpose(1, 2).contiguous())  # [2,H,4H]
        self.register_buffer("bias", (b_ih + b_hh).unsqueeze(1))  # [2,1,4H]

    def forward(self, x, valid=None):  # x [1,T,I] -> [1,T,2H]
        """`valid` (int tensor [1]) reverses only the first `valid` steps for the backward direction,
        so outputs at valid steps equal an unpadded run; padded steps follow and are discarded."""
        seq = x[0]
        if valid is None:
            backward = seq.flip(0)
            restore = None
        else:
            t = torch.arange(seq.shape[0])
            restore = torch.where(t < valid, valid - 1 - t, t)  # an involution on [0, T)
            backward = torch.index_select(seq, 0, restore)
        both = torch.stack([seq, backward])  # [2,T,I]
        gates_in = torch.bmm(both, self.w_ih_t) + self.bias  # [2,T,4H]
        h0 = torch.zeros(2, 1, self.hidden, dtype=x.dtype)
        c0 = torch.zeros(2, 1, self.hidden, dtype=x.dtype)
        out0 = torch.zeros(2, seq.shape[0], self.hidden, dtype=x.dtype)
        hidden = self.hidden

        def cond_fn(t, h, c, out, gates, w_hh):
            return t < gates.shape[1]

        def body_fn(t, h, c, out, gates, w_hh):
            g = torch.index_select(gates, 1, t.view(1)) + torch.bmm(h, w_hh)  # [2,1,4H]
            i, f, cand, o = torch.split(g, hidden, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(cand)
            h = torch.sigmoid(o) * torch.tanh(c)
            index = t.view(1, 1, 1).expand(2, 1, hidden)
            return t + 1, h, c, out.scatter(1, index, h)

        _, _, _, out = torch.ops.higher_order.while_loop(
            cond_fn, body_fn, (torch.tensor(0), h0, c0, out0), (gates_in, self.w_hh_t))
        back = out[1].flip(0) if restore is None else torch.index_select(out[1], 0, restore)
        return torch.cat([out[0], back], dim=-1).unsqueeze(0)


def upsample_nearest(x, factor: int):  # [B,C,N] -> [B,C,N*factor]
    b, c, n = x.shape
    return x.unsqueeze(-1).expand(b, c, n, factor).reshape(b, c, n * factor)


def upsample_linear(x, factor: int):
    """F.interpolate(x, scale_factor=factor, mode='linear') reproducing PyTorch's CPU kernel.

    The kernel samples src = float32(1/factor) * (j + 0.5) - 0.5 for each output j,
    clamps negative coordinates to 0 and repeats the last input at the upper edge.
    """
    b, c, n = x.shape
    scale = float(torch.tensor(1.0 / factor, dtype=torch.float32))
    j = torch.arange(n * factor, dtype=torch.float32)
    src = torch.clamp(scale * (j + 0.5) - 0.5, min=0.0)
    i0 = torch.floor(src)
    lam1 = torch.clamp(src - i0, 0.0, 1.0)
    lam0 = 1.0 - lam1
    i0 = i0.to(torch.int64)
    i1 = torch.clamp(i0 + 1, max=n - 1)
    return torch.index_select(x, 2, i0) * lam0 + torch.index_select(x, 2, i1) * lam1


class ExactSTFT(nn.Module):
    """torch.stft/istft(center=True, reflect, hann periodic, onesided) without complex ops."""

    def __init__(self, n_fft=N_FFT, hop=HOP):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        bins = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(bins, dtype=torch.float64)
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        # Twiddles from the exact residue k*n mod n_fft, with exact zeros/ones at quarter
        # turns. The FFT yields exactly zero imaginary DC/Nyquist parts; a 1e-17 residue
        # would flip atan2 between +pi and -pi there, and phase is a decoder feature.
        residue = torch.outer(k, n).remainder(n_fft)
        angle = 2 * math.pi * residue / n_fft  # [bins,n_fft]
        cos, sin = torch.cos(angle), torch.sin(angle)
        quarter = residue.remainder(n_fft // 4) == 0
        cos = torch.where(quarter, cos.round(), cos)
        sin = torch.where(quarter, sin.round(), sin)
        self.register_buffer("fwd_re", (cos * window).float().unsqueeze(1))
        self.register_buffer("fwd_im", (-sin * window).float().unsqueeze(1))
        scale = torch.full((bins, 1), 2.0, dtype=torch.float64)
        scale[0] = 1.0
        scale[-1] = 1.0  # n_fft even: Nyquist bin counted once
        self.register_buffer("inv_re", (scale * cos * window / n_fft).float().unsqueeze(1))
        self.register_buffer("inv_im", (-scale * sin * window / n_fft).float().unsqueeze(1))
        self.register_buffer("window_sq", (window ** 2).float().view(1, 1, n_fft))

    def transform(self, x):  # [1,L] -> magnitude, phase [1,bins,L/hop+1]
        pad = self.n_fft // 2
        x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect")
        re = F.conv1d(x, self.fwd_re, stride=self.hop)
        im = F.conv1d(x, self.fwd_im, stride=self.hop)
        return torch.sqrt(re * re + im * im), torch.atan2(im, re)

    def inverse(self, magnitude, phase):  # -> [1,1,hop*(frames-1)]
        re = magnitude * torch.cos(phase)
        im = magnitude * torch.sin(phase)
        y = F.conv_transpose1d(re, self.inv_re, stride=self.hop) + F.conv_transpose1d(im, self.inv_im, stride=self.hop)
        ones = torch.ones_like(magnitude[:, :1, :])
        envelope = F.conv_transpose1d(ones, self.window_sq, stride=self.hop)
        y = y / envelope
        pad = self.n_fft // 2
        return y[..., pad:y.shape[-1] - pad]


# ---------------------------------------------------------------------------
# Export graphs (built from a loaded KModel; weights shared, upstream forwards reimplemented)


def _strip_weight_norm(module: nn.Module):
    for m in module.modules():
        try:
            torch.nn.utils.remove_weight_norm(m)
        except ValueError:
            pass


_RSQRT2 = float(torch.rsqrt(torch.tensor(2.0)))  # upstream: torch.rsqrt(torch.tensor(2)) (int input)


def _adain_block_forward(self, x, s):
    return (self._residual(x, s) + self._shortcut(x)) * _RSQRT2


class AdaLayerNormX(nn.Module):
    def __init__(self, ada):
        super().__init__()
        self.ada = ada

    def forward(self, x, s):  # x [1,T,C] (channels last); mirrors upstream AdaLayerNorm
        h = self.ada.fc(s).view(s.shape[0], -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)  # [1,1,C]
        x = F.layer_norm(x, (self.ada.channels,), eps=self.ada.eps)
        return (1 + gamma) * x + beta


class DurationsGraph(nn.Module):
    def __init__(self, kmodel):
        super().__init__()
        self.bert = kmodel.bert
        self.bert_encoder = kmodel.bert_encoder
        enc = kmodel.predictor.text_encoder
        self.dur_blocks = nn.ModuleList()
        for block in enc.lstms:
            self.dur_blocks.append(LoopLSTM(block) if isinstance(block, nn.LSTM) else AdaLayerNormX(block))
        self.pred_lstm = LoopLSTM(kmodel.predictor.lstm)
        self.duration_proj = kmodel.predictor.duration_proj
        te = kmodel.text_encoder
        self.embedding = te.embedding
        self.cnn = te.cnn
        self.text_lstm = LoopLSTM(te.lstm)
        _strip_weight_norm(self)

    def forward(self, input_ids, style):
        ids = input_ids.to(torch.int64)
        bert = self.bert(ids, attention_mask=torch.ones_like(ids))
        d_en = self.bert_encoder(bert)  # [1,T,512] (upstream transposes to channels-first)
        s = style[:, 128:]
        t = ids.shape[1]
        s_rows = s.unsqueeze(1).expand(1, t, s.shape[-1])  # [1,T,128]
        x = torch.cat([d_en, s_rows], dim=-1)  # [1,T,640]
        for block in self.dur_blocks:
            if isinstance(block, AdaLayerNormX):
                x = torch.cat([block(x, s), s_rows], dim=-1)
            else:
                x = block(x)
        d = x  # [1,T,640]
        duration = torch.sigmoid(self.duration_proj(self.pred_lstm(d))).sum(dim=-1)  # [1,T]
        e = self.embedding(ids).transpose(1, 2)  # [1,512,T]
        for c in self.cnn:
            e = c(e)
        t_en = self.text_lstm(e.transpose(1, 2)).transpose(1, 2)  # [1,512,T]
        return duration, d, t_en


class DecoderGraph(nn.Module):
    def __init__(self, kmodel):
        super().__init__()
        p = kmodel.predictor
        self.shared = LoopLSTM(p.shared)
        self.F0, self.N, self.F0_proj, self.N_proj = p.F0, p.N, p.F0_proj, p.N_proj
        self.decoder = kmodel.decoder
        g = self.decoder.generator
        self.generator = g
        self.stft = ExactSTFT(g.post_n_fft, HOP)
        for blocks in (self.F0, self.N, self.decoder.decode, [self.decoder.encode]):
            for block in blocks:
                if block.upsample_type != "none":
                    block.upsample.forward = lambda x: upsample_nearest(x, 2)
                block.forward = types.MethodType(_adain_block_forward, block)
        _strip_weight_norm(self)
        sine = g.m_source.l_sin_gen
        assert sine.harmonic_num == HARMONICS - 1 and sine.upsample_scale == 300 and not sine.flag_for_pulse
        self.register_buffer("harmonics", torch.arange(1, HARMONICS + 1, dtype=torch.float32).view(1, 1, HARMONICS))

    def source(self, f0_curve, noise):
        """SineGen + SourceModuleHnNSF with provided standard-normal noise.

        f0_curve [1,2F] at twice the frame rate; noise [1,600F,9]. Returns har [1,600F].
        """
        sine = self.generator.m_source.l_sin_gen
        scale = sine.upsample_scale  # 300
        f0 = f0_curve.unsqueeze(-1)  # [1,2F,1] (nearest-upsampled samples are equal within a block)
        cycles = f0 * self.harmonics / sine.sampling_rate
        # == upstream `% 1` (floor modulo) after its exact downsampling; remainder.Scalar is not lowered.
        rad = cycles - torch.floor(cycles)  # [1,2F,9]
        phase = torch.cumsum(rad, dim=1) * 2 * math.pi
        phase = upsample_linear(phase.transpose(1, 2) * scale, scale).transpose(1, 2)  # [1,L,9]
        sines = torch.sin(phase) * sine.sine_amp
        uv = (f0 > sine.voiced_threshold).to(torch.float32)  # [1,2F,1]
        uv = upsample_nearest(uv.transpose(1, 2), scale).transpose(1, 2)  # [1,L,1]
        noise_amp = uv * sine.noise_std + (1 - uv) * sine.sine_amp / 3
        waves = sines * uv + noise_amp * noise
        m = self.generator.m_source
        return m.l_tanh(m.l_linear(waves)).transpose(1, 2).squeeze(1)  # [1,L]

    def forward(self, en, asr, style, noise):
        s = style[:, 128:]
        x = self.shared(en.transpose(-1, -2))  # [1,F,512]
        f0 = x.transpose(-1, -2)
        for block in self.F0:
            f0 = block(f0, s)
        f0_curve = self.F0_proj(f0).squeeze(1)  # [1,2F]
        n = x.transpose(-1, -2)
        for block in self.N:
            n = block(n, s)
        n_curve = self.N_proj(n).squeeze(1)

        dec = self.decoder
        ref = style[:, :128]
        f0d = dec.F0_conv(f0_curve.unsqueeze(1))
        nd = dec.N_conv(n_curve.unsqueeze(1))
        h = dec.encode(torch.cat([asr, f0d, nd], dim=1), ref)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                h = torch.cat([h, asr_res, f0d, nd], dim=1)
            h = block(h, ref)
            if block.upsample_type != "none":
                res = False

        g = self.generator
        har_source = self.source(f0_curve, noise.reshape(1, -1, HARMONICS))
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(g.num_upsamples):
            h = F.leaky_relu(h, negative_slope=0.1)
            x_source = g.noise_res[i](g.noise_convs[i](har), ref)
            h = g.ups[i](h)
            if i == g.num_upsamples - 1:
                h = g.reflection_pad(h)
            h = h + x_source
            xs = None
            for j in range(g.num_kernels):
                r = g.resblocks[i * g.num_kernels + j](h, ref)
                xs = r if xs is None else xs + r
            h = xs / g.num_kernels
        h = F.leaky_relu(h)
        h = g.conv_post(h)
        bins = g.post_n_fft // 2 + 1
        audio = self.stft.inverse(torch.exp(h[:, :bins, :]), torch.sin(h[:, bins:, :]))
        return audio.squeeze(1)  # [1,600F]


def _prefix_mask(length, valid, dtype=torch.float32):
    """[1,1,length] float mask, 1 for positions < valid (int tensor [1])."""
    # unsqueeze, not view(1, 1, length): the reshape form fails MPS type inference on the GPU.
    return (torch.arange(length) < valid).to(dtype).unsqueeze(0).unsqueeze(0)


def _masked_adain(norm_module, x, s, m, n):
    """AdaIN1d with InstanceNorm statistics over the first n positions only (m is their mask)."""
    h = norm_module.fc(s).view(s.shape[0], -1, 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    norm = norm_module.norm
    mean = (x * m).sum(dim=-1, keepdim=True) / n
    centered = (x - mean) * m
    var = (centered * centered).sum(dim=-1, keepdim=True) / n
    x_hat = (x - mean) / torch.sqrt(var + norm.eps)
    if norm.affine:
        x_hat = x_hat * norm.weight.view(1, -1, 1) + norm.bias.view(1, -1, 1)
    return (1 + gamma) * x_hat + beta


def _masked_adain_block(block, x, s, m_in, n_in, m_out, n_out):
    """Upstream AdainResBlk1d with every conv input masked like the unpadded run's zero padding."""
    r = block.actv(_masked_adain(block.norm1, x, s, m_in, n_in)) * m_in
    r = block.pool(r)
    r = block.conv1(r * m_out)
    r = block.actv(_masked_adain(block.norm2, r, s, m_out, n_out)) * m_out
    r = block.conv2(r)
    shortcut = block.upsample(x)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    return (r + shortcut) * _RSQRT2


def _masked_resblock(block, x, s, m, n):
    """Upstream AdaINResBlock1 (Snake1D) with masked statistics and conv inputs."""
    for c1, c2, n1, n2, a1, a2 in zip(block.convs1, block.convs2, block.adain1, block.adain2, block.alpha1, block.alpha2):
        xt = _masked_adain(n1, x, s, m, n)
        xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)
        xt = c1(xt * m)
        xt = _masked_adain(n2, xt, s, m, n)
        xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
        xt = c2(xt * m)
        x = xt + x
    return x


class BucketedDecoderGraph(DecoderGraph):
    """The decoder with an extra `frames` input (int32 [1]): inputs may be zero-padded to a fixed
    frame bucket F_b >= frames, and the first 600*frames output samples equal an unpadded run.

    Every time-global or edge-sensitive operation is restricted to the valid prefix at its own
    resolution (F, 2F, 20F, 120F+1 frames; 600F samples): InstanceNorm statistics, conv inputs
    (masked like zero padding), the reverse LSTM direction, the F0 phase interpolation edge, the
    harmonic STFT's end reflection and the iSTFT frames and window envelope.
    """

    def source(self, f0_curve, noise, valid_f0=None):
        if valid_f0 is None:
            return super().source(f0_curve, noise)
        sine = self.generator.m_source.l_sin_gen
        scale = sine.upsample_scale
        f0 = f0_curve.unsqueeze(-1)
        cycles = f0 * self.harmonics / sine.sampling_rate
        rad = cycles - torch.floor(cycles)
        phase = torch.cumsum(rad, dim=1) * 2 * math.pi
        x = phase.transpose(1, 2) * scale  # [1,9,2F]
        # The unpadded interpolation repeats the last valid frame at its upper edge. Hold that value
        # over the padding (x*1 + last*0 keeps valid frames bit-exact) so the interpolation indices
        # stay static per shape; only a [1,9,1] gather depends on the runtime frame count.
        hold = _prefix_mask(x.shape[-1], valid_f0, x.dtype)
        last = torch.index_select(x, 2, (valid_f0 - 1).view(1))
        phase = upsample_linear(x * hold + last * (1 - hold), scale).transpose(1, 2)
        sines = torch.sin(phase) * sine.sine_amp
        uv = (f0 > sine.voiced_threshold).to(torch.float32)
        uv = upsample_nearest(uv.transpose(1, 2), scale).transpose(1, 2)
        noise_amp = uv * sine.noise_std + (1 - uv) * sine.sine_amp / 3
        waves = sines * uv + noise_amp * noise
        m = self.generator.m_source
        return m.l_tanh(m.l_linear(waves)).transpose(1, 2).squeeze(1)

    def forward(self, en, asr, style, noise, frames):
        fb = en.shape[-1]
        valid = frames.to(torch.int64).view(1)
        dt = en.dtype
        m1, m2 = _prefix_mask(fb, valid, dt), _prefix_mask(2 * fb, 2 * valid, dt)
        m20, m120 = _prefix_mask(20 * fb, 20 * valid, dt), _prefix_mask(120 * fb + 1, 120 * valid + 1, dt)
        n1, n2, n20, n120 = (valid.to(dt) * k for k in (1, 2, 20, 120))
        n120 = n120 + 1
        en, asr = en * m1, asr * m1
        s = style[:, 128:]
        x = self.shared(en.transpose(-1, -2), valid)
        f0 = x.transpose(-1, -2)
        n = f0
        for blocks, curve in ((self.F0, "f0"), (self.N, "n")):
            h = x.transpose(-1, -2)
            level = 1
            for block in blocks:
                up = block.upsample_type != "none"
                (mi, ni), (mo, no) = ((m1, n1) if level == 1 else (m2, n2)), ((m2, n2) if up or level == 2 else (m1, n1))
                h = _masked_adain_block(block, h, s, mi, ni, mo, no)
                level = 2 if up else level
            if curve == "f0":
                f0 = self.F0_proj(h * m2).squeeze(1)
            else:
                n = self.N_proj(h * m2).squeeze(1)
        f0_curve, n_curve = f0 * m2[:, 0], n * m2[:, 0]

        dec = self.decoder
        ref = style[:, :128]
        f0d = dec.F0_conv(f0_curve.unsqueeze(1)) * m1
        nd = dec.N_conv(n_curve.unsqueeze(1)) * m1
        h = _masked_adain_block(dec.encode, torch.cat([asr, f0d, nd], dim=1), ref, m1, n1, m1, n1)
        asr_res = dec.asr_res(asr)
        res, level = True, 1
        for block in dec.decode:
            if res:
                h = torch.cat([h, asr_res, f0d, nd], dim=1)
            up = block.upsample_type != "none"
            (mi, ni), (mo, no) = ((m1, n1) if level == 1 else (m2, n2)), ((m2, n2) if up or level == 2 else (m1, n1))
            h = _masked_adain_block(block, h, ref, mi, ni, mo, no)
            if up:
                res, level = False, 2

        g = self.generator
        har_source = self.source(f0_curve, noise.reshape(1, -1, HARMONICS), 2 * valid)
        # Reflect the harmonic source at the valid end, as the unpadded STFT's reflect padding does:
        # write har[Lv-2-j] to position Lv+j for j < 10. Only these ten samples use runtime indices;
        # the 10-sample extension absorbs the writes when the input exactly fills its bucket.
        length = har_source.shape[-1]
        pad = self.stft.n_fft // 2
        valid_samples = valid * SAMPLES_PER_FRAME
        offsets = torch.arange(pad)
        tail = torch.index_select(har_source, 1, valid_samples - 2 - offsets)
        extended = torch.cat([har_source, torch.zeros(1, pad, dtype=har_source.dtype)], dim=1)
        har_source = extended.scatter(1, (valid_samples + offsets).view(1, pad), tail.view(1, pad))[:, :length]
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1) * m120
        masks = [(m20, n20), (m120, n120)]
        for i in range(g.num_upsamples):
            h = F.leaky_relu(h, negative_slope=0.1) * (m2 if i == 0 else m20)
            mi, ni = masks[i]
            x_source = _masked_resblock(g.noise_res[i], g.noise_convs[i](har) * mi, ref, mi, ni)
            h = g.ups[i](h)
            if i == g.num_upsamples - 1:
                h = g.reflection_pad(h)
            h = h + x_source
            xs = None
            for j in range(g.num_kernels):
                r = _masked_resblock(g.resblocks[i * g.num_kernels + j], h, ref, mi, ni)
                xs = r if xs is None else xs + r
            h = xs / g.num_kernels
        h = F.leaky_relu(h) * m120
        h = g.conv_post(h)
        bins = g.post_n_fft // 2 + 1
        magnitude = torch.exp(h[:, :bins, :]) * m120
        phase = torch.sin(h[:, bins:, :])
        st = self.stft
        re, im = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        y = F.conv_transpose1d(re, st.inv_re, stride=st.hop) + F.conv_transpose1d(im, st.inv_im, stride=st.hop)
        envelope = F.conv_transpose1d(m120, st.window_sq, stride=st.hop)
        y = y / torch.clamp(envelope, min=1e-11)  # beyond the valid frames the envelope is zero
        pad = st.n_fft // 2
        return y[..., pad:y.shape[-1] - pad].squeeze(1)  # [1,600F_b]; samples >= 600*frames are padding


# ---------------------------------------------------------------------------
# Host steps shared with the Swift implementation


def round_durations(duration_sums: torch.Tensor, speed: float) -> torch.Tensor:
    return torch.round(duration_sums / speed).clamp(min=1).long().squeeze(0)


def align(d, t_en, durations):
    """Gather equivalent of upstream's one-hot alignment matmul."""
    index = torch.repeat_interleave(torch.arange(durations.shape[0]), durations)
    en = d[0, index, :].transpose(0, 1).unsqueeze(0)  # [1,640,F]
    asr = t_en[0, :, index].unsqueeze(0)  # [1,512,F]
    return en, asr


def splitmix_normal(seed: int, count: int):
    """Port of AetherSpeech's SplitMix64 + double Box-Muller generator (same as KreaNoise)."""
    import numpy as np

    mask = (1 << 64) - 1
    state = seed & mask
    out = np.empty(count, dtype=np.float32)
    i = 0

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & mask
        v = state
        v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & mask
        v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & mask
        return v ^ (v >> 31)

    while i < count:
        u1 = ((nxt() >> 11) + 1) / 9007199254740992
        u2 = (nxt() >> 11) / 9007199254740992
        radius = math.sqrt(-2 * math.log(u1))
        angle = 2 * math.pi * u2
        out[i] = radius * math.cos(angle)
        i += 1
        if i < count:
            out[i] = radius * math.sin(angle)
            i += 1
    return out


def tokenize(vocab: dict, phonemes: str):
    ids = []
    for ch in phonemes:  # upstream iterates Unicode scalars and drops unknown symbols
        if ch not in vocab:
            raise ValueError(f"phoneme {ch!r} (U+{ord(ch):04X}) is not in the Kokoro vocabulary")
        ids.append(vocab[ch])
    if not 1 <= len(ids) <= MAX_PHONEMES:
        raise ValueError("phoneme count must be 1...510")
    return [0, *ids, 0]
