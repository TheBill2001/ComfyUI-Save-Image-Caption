from . import nodes

NODE_CLASS_MAPPINGS: dict[str, nodes.BaseNode] = {
    nodes.LoadImageBatch.NODE_CLASS_MAPPING: nodes.LoadImageBatch,
    nodes.SaveImageCaptionBatch.NODE_CLASS_MAPPING: nodes.SaveImageCaptionBatch,
    # nodes.SaveImageCaptionBatchList.NODE_CLASS_MAPPING: nodes.SaveImageCaptionBatchList,
}

NODE_DISPLAY_NAME_MAPPINGS = {node_class: node.NODE_DISPLAY_NAME_MAPPING for node_class, node in NODE_CLASS_MAPPINGS.items()}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']