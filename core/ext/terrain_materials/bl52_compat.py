"""A shim for Infinigen 1.x nodes (written for Blender 4.2) in Blender 5.2: for eight nodes the sockets became "dynamic" (one Value
socket instead of five typed ones), and the old socket numbers from the transpiled code point to the wrong place.
We translate the old number → the socket name on inputs (infer_input_socket) and on outputs (a wrapper node with .outputs[number]).
Install it after ig_env and before importing the generators: import bl52_compat; bl52_compat.install()."""
import bpy

# The Blender 4.2 socket layout (number → name); in 5.2 we look for an enabled socket with this name
OLD_IN = {
    "FunctionNodeCompare": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "C", "Angle", "Epsilon"],
    "FunctionNodeRandomValue": ["Min", "Max", "Min", "Max", "Min", "Max", "Probability", "ID", "Seed"],
    "GeometryNodeRaycast": ["Target Geometry", "Attribute", "Attribute", "Attribute", "Attribute", "Attribute",
                            "Source Position", "Ray Direction", "Ray Length"],
    "GeometryNodeAccumulateField": ["Value", "Value", "Value", "Group ID"],
    "GeometryNodeFieldAtIndex": ["Index", "Value", "Value", "Value", "Value", "Value"],
    "GeometryNodeSampleNearestSurface": ["Mesh", "Value", "Value", "Value", "Value", "Value", "Group ID",
                                         "Sample Position", "Sample Group ID"],
}
OLD_IN_FN = {   # layouts where the middle is a Value of any length
    "GeometryNodeSampleIndex": lambda i: "Geometry" if i == 0 else "Value",
    "GeometryNodeStoreNamedAttribute": lambda i: ["Geometry", "Selection", "Name"][i] if i < 3 else "Value",
}
RENAME_IN = {"Group Index": "Group ID"}
OLD_OUT = {
    "FunctionNodeRandomValue": lambda i: "Value",
    "GeometryNodeSampleIndex": lambda i: "Value",
    "GeometryNodeSampleNearestSurface": lambda i: "Value",
    "GeometryNodeFieldAtIndex": lambda i: "Value",
    "GeometryNodeAccumulateField": lambda i: ["Leading", "Trailing", "Total"][min(i // 3, 2)],
    "GeometryNodeRaycast": lambda i: ["Is Hit", "Hit Position", "Hit Normal", "Hit Distance"][i] if i < 4 else "Attribute",
}
STATS = {"in": 0, "out": 0}


def _find(coll, name):
    for s in coll:
        if s.name == name and s.enabled:
            return s
    return coll[name]


class _Outs:
    def __init__(self, node):
        self._node = node

    def __getitem__(self, k):
        f = OLD_OUT.get(self._node.bl_idname)
        if isinstance(k, int) and f is not None:
            STATS["out"] += 1
            return _find(self._node.outputs, f(k))
        return self._node.outputs[k]

    def __iter__(self):
        return iter(self._node.outputs)

    def __len__(self):
        return len(self._node.outputs)

    def __getattr__(self, k):
        return getattr(self._node.outputs, k)


class NodeProxy:
    """A node that translates old output numbers. Everything else goes to the real node."""
    __slots__ = ("_n",)

    def __init__(self, n):
        object.__setattr__(self, "_n", n)

    @property
    def outputs(self):
        return _Outs(self._n)

    def __getattr__(self, k):
        return getattr(self._n, k)

    def __setattr__(self, k, v):
        setattr(self._n, k, v)

    def __eq__(self, o):
        return self._n == (o._n if isinstance(o, NodeProxy) else o)

    def __hash__(self):
        return hash(self._n)


def unwrap(x):
    return x._n if isinstance(x, NodeProxy) else x


def _holder(mod, base):
    p = mod.properties
    for coll in (p.inputs, p.outputs):
        if base in coll.bl_rna.properties.keys():
            return getattr(coll, base)
    return None


def _split(k):
    for suf, f in (("_use_attribute", "use"), ("_attribute_name", "attribute_name")):
        if isinstance(k, str) and k.endswith(suf):
            return k[: -len(suf)], f
    return k, "value"


def _install_modifier_items():
    """5.x: node modifier inputs are mod.properties.inputs.Socket_N.{value,type,attribute_name}, not mod["Socket_N"]."""
    T = bpy.types.NodesModifier
    orig_set, orig_get = T.__setitem__, T.__getitem__

    def setitem(self, k, v):
        base, field = _split(k)
        h = _holder(self, base) if isinstance(base, str) else None
        if h is None:
            return orig_set(self, k, v)
        STATS["mod"] = STATS.get("mod", 0) + 1
        if field == "value":
            h.value = v
        elif field == "use":
            h.type = "ATTRIBUTE" if v else "VALUE"
        else:
            h.attribute_name = v

    def getitem(self, k):
        base, field = _split(k)
        h = _holder(self, base) if isinstance(base, str) else None
        if h is None:
            return orig_get(self, k)
        if field == "value":
            return h.value
        if field == "use":
            return int(h.type == "ATTRIBUTE")
        return h.attribute_name

    T.__setitem__, T.__getitem__ = setitem, getitem


def install():
    _install_modifier_items()
    from infinigen.core.nodes import node_wrangler as NW
    from infinigen.core.nodes import utils as U

    orig_in, orig_out, orig_isnode = U.infer_input_socket, U.infer_output_socket, U.isnode

    def infer_input_socket(node, name):
        node = unwrap(node)
        if isinstance(name, str) and name in RENAME_IN and name not in node.inputs:
            name = RENAME_IN[name]
        if isinstance(name, int):
            idn = node.bl_idname
            if idn in OLD_IN and name < len(OLD_IN[idn]):
                STATS["in"] += 1
                return _find(node.inputs, OLD_IN[idn][name])
            if idn in OLD_IN_FN:
                STATS["in"] += 1
                return _find(node.inputs, OLD_IN_FN[idn](name))
        return orig_in(node, name)

    def infer_output_socket(item):
        if isinstance(item, tuple) and item and isinstance(item[0], NodeProxy):
            item = (item[0]._n,) + item[1:]
        return orig_out(unwrap(item))

    def isnode(x):
        return orig_isnode(unwrap(x))

    U.infer_input_socket, U.infer_output_socket, U.isnode = infer_input_socket, infer_output_socket, isnode
    NW.infer_input_socket, NW.infer_output_socket = infer_input_socket, infer_output_socket

    orig_new = NW.NodeWrangler.new_node
    orig_connect = NW.NodeWrangler.connect_input

    def new_node(self, node_type, input_args=None, attrs=None, input_kwargs=None, *a, **k):
        if node_type == "GeometryNodeMeshBoolean":       # 4.2: the EXACT solver by default (in 5.x it differs, there are no Self Intersection sockets)
            attrs = dict(attrs or {})
            attrs.setdefault("solver", "EXACT")
        if input_args:
            input_args = [unwrap(x) for x in input_args]
        if input_kwargs:
            input_kwargs = {kk: unwrap(v) for kk, v in input_kwargs.items()}
        n = orig_new(self, node_type, input_args, attrs, input_kwargs, *a, **k)
        if isinstance(n, bpy.types.Node) and n.bl_idname in OLD_OUT:
            return NodeProxy(n)
        return n

    def connect_input(self, sock, item):
        if isinstance(item, list):
            item = [unwrap(x) for x in item]
        return orig_connect(self, sock, unwrap(item))

    NW.NodeWrangler.new_node = new_node
    NW.NodeWrangler.connect_input = connect_input
    return STATS
