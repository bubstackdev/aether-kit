"""Mechanical weight lifting for original Torch modules, including root parameters."""
def parameter_input_graph(module, inputs):
    import torch
    root = torch.export.export(module, inputs).module()
    parameters = dict(root.named_parameters())
    last = next(n for n in reversed(list(root.graph.nodes)) if n.op == "placeholder")
    for i, name in enumerate(parameters):
        nodes = [n for n in root.graph.nodes if n.op == "get_attr" and n.target == name]
        assert nodes, name
        with root.graph.inserting_after(last):
            last = root.graph.placeholder(f"w{i}")
        last.meta = nodes[0].meta.copy()
        for node in nodes:
            node.replace_all_uses_with(last)
            root.graph.erase_node(node)
    root.graph.eliminate_dead_code()
    output = next(n for n in root.graph.nodes if n.op == "output")
    assert isinstance(output.args[0], tuple) and len(output.args[0]) == 1
    output.args = (output.args[0][0],)
    root.graph.set_codegen(torch.fx.graph.CodeGen())
    # Constructing a fresh root copies only referenced attributes. Pruning
    # submodules alone leaves unused parameters registered directly on root.
    graph = torch.fx.GraphModule(root, root.graph)
    graph.graph.lint()
    assert not dict(graph.named_parameters())
    return graph, parameters
