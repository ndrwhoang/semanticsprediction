def _index_node_logits(self, raw_logits, node_labels: List[Dict]):
    node_outputs = []
    for i_sample, node_label in enumerate(node_labels):
        node_output = [torch.mean(raw_logits[i_sample, idx, :], dim=0) for idx in node_label.keys()]
        node_output = torch.stack(node_output, dim=0)
        node_outputs.append(node_output)              
    node_outputs = pad_sequence(node_outputs, batch_first=True)
    
    return node_outputs
    
def _index_edge_logits(self, raw_logits, edge_labels: List[Dict]):
    edge_outputs = []
    for i_sample, edge_id_pair in enumerate(edge_labels):
        edge_output = [torch.cat((torch.mean(raw_logits[i_sample, idx[0], :], dim=0), torch.mean(raw_logits[i_sample, idx[1], :], dim=0))) for idx in edge_id_pair.keys()]
        edge_output = torch.stack(edge_output, dim=0)
        edge_outputs.append(edge_output)
    edge_outputs = pad_sequence(edge_outputs, batch_first=True)
    
    return edge_outputs