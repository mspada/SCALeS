"""
Diagnostic reports of SupplyChain: LP constraint reports (PDF) and time-series tables (CSV) of flows and of the
solver's slack variables GENERATION, ABSORPTION and BACKORDER.
"""

import csv
import os
from string import ascii_uppercase

from .. import config
from .._utils import natural_sort_key


class DiagnosticsMixin:
    """Methods of SupplyChain that report on the (solved) dynamic network."""

    def lp_node_diagnostic(self, label: str, t: int, filename: str = None) -> None:
        """
        Creates a PDF file listing all constraints (equalities and inequalities)
        for the involved nodes in the time neighborhood of the specified backbone
        node and time step t. Optimization status is reported. It includes additionally 
        optimal flows if they are computed. If optimization fails the method lists 
        lp information without optimal flows and provides status.
        Default filename is results/diagnostics/lp/constraints_<label>_t<t>.pdf.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
        except ImportError as e:
            raise ImportError("lp_node_diagnostic needs reportlab: pip install reportlab") from e

        subgraph_nodes, t_min, t_max = self._node_time_neighbourhood(label, t)

        # Sort the involved nodes to have a deterministic order in the report
        # Sort by time t first, and then naturally by name
        sorted_nodes = sorted(list(subgraph_nodes), key=lambda x: (x[1], natural_sort_key(x[0])))

        # Prepare default filename if not provided
        if not filename:
            filename = str(config.results_dir("diagnostics", "lp") / f"constraints_{label}_t{t}.pdf")
        elif not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"

        # Create output directory if needed
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Build PDF doc
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        
        styles = getSampleStyleSheet()
        
        # Define some custom styles for the report
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1A365D'),
            spaceAfter=10
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#4A5568'),
            spaceAfter=20
        )
        node_header_style = ParagraphStyle(
            'NodeHeader',
            parent=styles['Heading3'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#2D3748'),
            spaceBefore=8,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#2D3748')
        )
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#1A202C'),
            backColor=colors.HexColor('#EDF2F7'),
            borderColor=colors.HexColor('#CBD5E0'),
            borderWidth=0.5,
            borderPadding=4,
            spaceBefore=4,
            spaceAfter=4
        )

        story = []
        
        # 1. Header Information
        story.append(Paragraph("Supply Chain Optimization Diagnostic Constraints", title_style))
        story.append(Paragraph(
            f"<b>Target Backbone Node:</b> {label} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Target Time:</b> {t} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Time Window:</b> t={t_min} to t={t_max} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Total Involved Nodes:</b> {len(sorted_nodes)}",
            subtitle_style
        ))
        
        # 2. Optimization Status
        opt_status = "<b>FOUND (Solved)</b>" if self._optimum_found else "<b>NOT FOUND (Not Solved Yet)</b>"
        story.append(Paragraph(f"<b>Optimization Status:</b> {opt_status}", body_style))
        story.append(Spacer(1, 10))

        # 3. Diagnostic description
        desc_text = (
            "This report lists all flow balance equations (equalities), capacity limits (inequalities), "
            "and variable bounds (non-negativity) for the nodes in the local time neighborhood of the "
            "specified target node. The equations are presented in symbolic algebraic form, numerical parameter form, "
            "and - if the model is optimized - with their actual optimal flow and variable values plugged in."
        )
        story.append(Paragraph(desc_text, body_style))
        story.append(Spacer(1, 15))

        # Make sure production capabilities are analyzed
        self._checkProductionCapabilities()

        # 4. List constraints for each involved node
        for node in sorted_nodes:
            node_name, node_t = node
            node_data = self._network.nodes[node]
            node_type = node_data.get('type', 'UNKNOWN')
            
            # Retrieve node parameters
            supply_val = node_data.get('supply', 0.0)
            demand_val = node_data.get('demand', 0.0)
            
            # Determine rule and capability
            can_produce = self._canProduce(node_name, node_t)
            n_rules = len(node_data['rule']) if can_produce else 1
            
            # Node header block
            node_info_str = f"<b>Node:</b> {node_name} at t={node_t} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Type:</b> {node_type}"
            if supply_val > 0:
                node_info_str += f" &nbsp;&nbsp;|&nbsp;&nbsp; <b>Supply:</b> {supply_val}"
            if demand_val > 0:
                node_info_str += f" &nbsp;&nbsp;|&nbsp;&nbsp; <b>Demand:</b> {demand_val}"
            
            node_story = []
            node_story.append(Paragraph(node_info_str, node_header_style))
            
            # Let's collect all predecessors and successors
            preds = list(self._network.predecessors(node))
            succs = list(self._network.successors(node))
            
            # Build Flow Rule Constraint(s)
            for k in range(n_rules):
                # Retrieve input coefficients
                coeffs = {pred: 0.0 for pred in preds}
                if node_type in ('PRODUCTION', 'PRODUCTION_INPUT') and node_data.get('prod_pred'):
                    prod_pred = node_data['prod_pred']
                    rule_dict = node_data['rule']
                    prod_keys = list(prod_pred.keys())
                    j = prod_keys[k]
                    for pred in node_data.get('supp_pred', []):
                        coeffs[pred] = 1.0
                    for tj in prod_pred[j]:
                        pred = (j, tj)
                        if pred in coeffs:
                            coeffs[pred] = 1.0 / rule_dict[j]
                elif node_type in ('PRODUCTION', 'PRODUCTION_INPUT'):
                    for pred in node_data.get('supp_pred', []):
                        coeffs[pred] = 1.0
                else:
                    for pred in preds:
                        coeffs[pred] = 1.0
                        
                # 1) Symbolic formulation
                lhs_terms = []
                for p in preds:
                    c = coeffs[p]
                    if c != 0.0:
                        c_str = f"{c:.2f} * " if c != 1.0 else ""
                        lhs_terms.append(f"{c_str}FLOW[{p[0]},{p[1]} &rarr; {node_name},{node_t}]")
                if supply_val > 0:
                    lhs_terms.append(f"SUPPLY[{node_name},{node_t}]")
                lhs_terms.append(f"GENERATION[{node_name},{node_t}]")
                lhs_terms.append(f"BACKORDER[{node_name},{node_t}]")
                
                rhs_terms = []
                for s in succs:
                    rhs_terms.append(f"FLOW[{node_name},{node_t} &rarr; {s[0]},{s[1]}]")
                if demand_val > 0:
                    rhs_terms.append(f"DEMAND[{node_name},{node_t}]")
                rhs_terms.append(f"ABSORPTION[{node_name},{node_t}]")
                if node_t > 0:
                    rhs_terms.append(f"BACKORDER[{node_name},{node_t-1}]")
                    
                lhs_symbolic = " + ".join(lhs_terms)
                rhs_symbolic = " + ".join(rhs_terms)
                
                # 2) Numeric parameter formulation
                lhs_num_terms = []
                for p in preds:
                    c = coeffs[p]
                    if c != 0.0:
                        c_str = f"{c:.2f} * " if c != 1.0 else ""
                        lhs_num_terms.append(f"{c_str}FLOW[{p[0]},{p[1]}]")
                if supply_val > 0:
                    lhs_num_terms.append(f"{supply_val:.1f}")
                lhs_num_terms.append(f"GENERATION")
                lhs_num_terms.append(f"BACKORDER")
                
                rhs_num_terms = []
                for s in succs:
                    rhs_num_terms.append(f"FLOW[{s[0]},{s[1]}]")
                if demand_val > 0:
                    rhs_num_terms.append(f"{demand_val:.1f}")
                rhs_num_terms.append(f"ABSORPTION")
                if node_t > 0:
                    rhs_num_terms.append(f"BACKORDER(t-1)")
                    
                lhs_numeric = " + ".join(lhs_num_terms)
                rhs_numeric = " + ".join(rhs_num_terms)
                
                rule_header = f"<b>Flow Conservation Equality (Rule {k}):</b>"
                node_story.append(Paragraph(rule_header, body_style))
                
                eq_text = f"<b>Symbolic:</b> {lhs_symbolic} = {rhs_symbolic}<br/>"
                eq_text += f"<b>Numeric:</b> {lhs_numeric} = {rhs_numeric}"
                
                # 3) Plug in optimal values if optimized
                if self._optimum_found:
                    lhs_opt_terms = []
                    lhs_opt_vals = []
                    for p in preds:
                        c = coeffs[p]
                        if c != 0.0:
                            f_val = self._network[p][node].get('flow', 0.0)
                            lhs_opt_terms.append(f"{c:.2f} * ({f_val:.1f})")
                            lhs_opt_vals.append(c * f_val)
                    if supply_val > 0:
                        lhs_opt_terms.append(f"{supply_val:.1f}")
                        lhs_opt_vals.append(supply_val)
                    
                    gen_val = node_data.get('generation', 0.0)
                    lhs_opt_terms.append(f"({gen_val:.1f})")
                    lhs_opt_vals.append(gen_val)
                    
                    bo_val = node_data.get('backorder', 0.0)
                    lhs_opt_terms.append(f"({bo_val:.1f})")
                    lhs_opt_vals.append(bo_val)
                    
                    rhs_opt_terms = []
                    rhs_opt_vals = []
                    for s in succs:
                        f_val = self._network[node][s].get('flow', 0.0)
                        rhs_opt_terms.append(f"({f_val:.1f})")
                        rhs_opt_vals.append(f_val)
                    if demand_val > 0:
                        rhs_opt_terms.append(f"{demand_val:.1f}")
                        rhs_opt_vals.append(demand_val)
                        
                    abs_val = node_data.get('absorption', 0.0)
                    rhs_opt_terms.append(f"({abs_val:.1f})")
                    rhs_opt_vals.append(abs_val)
                    
                    if node_t > 0:
                        prev_node_data = self._network.nodes[(node_name, node_t - 1)]
                        prev_bo_val = prev_node_data.get('backorder', 0.0)
                        rhs_opt_terms.append(f"({prev_bo_val:.1f})")
                        rhs_opt_vals.append(prev_bo_val)
                        
                    lhs_opt_sum = sum(lhs_opt_vals)
                    rhs_opt_sum = sum(rhs_opt_vals)
                    
                    lhs_opt_str = " + ".join(lhs_opt_terms)
                    rhs_opt_str = " + ".join(rhs_opt_terms)
                    
                    eq_text += f"<br/><b>Optimal Values:</b> {lhs_opt_str} = {rhs_opt_str} &nbsp;&rArr;&nbsp; <b>{lhs_opt_sum:.1f} = {rhs_opt_sum:.1f}</b>"
                
                node_story.append(Paragraph(eq_text, code_style))

            # Capacity Limits
            cap_lines = []
            for s in succs:
                edge_data = self._network[node][s]
                cap_val = edge_data.get('cap', float('inf'))
                cap_str = f"{cap_val:.0f}" if cap_val != float('inf') else "&infin;"
                
                is_storage = (s[0] == node_name)
                edge_desc = "Storage Cap" if is_storage else "Transport Cap"
                
                ineq_symbolic = f"FLOW[{node_name},{node_t} &rarr; {s[0]},{s[1]}] &le; {cap_str} ({edge_desc})"
                
                if self._optimum_found:
                    f_val = edge_data.get('flow', 0.0)
                    ineq_symbolic += f" &nbsp;&nbsp;|&nbsp;&nbsp; <b>Optimal:</b> {f_val:.1f} &le; {cap_str}"
                
                cap_lines.append(ineq_symbolic)
            
            if cap_lines:
                node_story.append(Paragraph("<b>Edge Capacity Inequalities:</b>", body_style))
                cap_text = "<br/>".join(cap_lines)
                node_story.append(Paragraph(cap_text, code_style))

            # Non-negativity Bounds
            bounds = [
                f"GENERATION[{node_name},{node_t}] &ge; 0",
                f"ABSORPTION[{node_name},{node_t}] &ge; 0",
                f"BACKORDER[{node_name},{node_t}] &ge; 0"
            ]
            for s in succs:
                bounds.append(f"FLOW[{node_name},{node_t} &rarr; {s[0]},{s[1]}] &ge; 0")
                
            node_story.append(Paragraph("<b>Non-negativity Bounds:</b>", body_style))
            bounds_text = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(bounds)
            node_story.append(Paragraph(bounds_text, code_style))
            
            story.append(KeepTogether(node_story))
            story.append(Spacer(1, 10))

        doc.build(story)

    def layer_flow_input(self, filename: str = None, layer: int = 0) -> None:
        """
        Generates a CSV table where rows correspond to discrete time up to duration-1.
        Columns are labeled with local input storages (I) and warehouses (W) of the specified layer.
        Values represent the total incoming amount of goods (from previous time steps,
        transports, and external supply) at that time. Columns are grouped consecutively
        by their corresponding production node.
        Default filename is results/diagnostics/flow/layer_<layer>_input.csv.
        """
        if filename is None:
            filename = str(config.results_dir("diagnostics", "flow") / f"layer_{layer}_input.csv")
        self._layer_flow_table(filename, layer, '(I')

    def layer_flow_output(self, filename: str = None, layer: int = 0) -> None:
        """
        Generates a CSV table where rows correspond to discrete time up to duration-1.
        Columns are labeled with local output storages (O) and warehouses (W) of the specified layer.
        Values represent the total incoming amount of goods (from previous time steps,
        productions, and external supply) at that time. Columns are grouped consecutively
        by their corresponding production node.
        Default filename is results/diagnostics/flow/layer_<layer>_output.csv.
        """
        if filename is None:
            filename = str(config.results_dir("diagnostics", "flow") / f"layer_{layer}_output.csv")
        self._layer_flow_table(filename, layer, '(O')

    def diagnostic_backorder(self, filename: str = None) -> None:
        """
        Generates a CSV table detailing the BACKORDER variable over time.
        Rows = Time, Columns = Nodes that are allowed to carry backorder debt (e.g. Storages, OEM).
        Default filename is results/diagnostics/time/backorder.csv.
        """
        self._variable_table(filename, 'backorder', 'backorders', self._non_production_node_names())

    def diagnostic_generation(self, filename: str = None) -> None:
        """
        Generates a CSV table detailing the GENERATION variable over time.
        This tracks emergency physical flows spawned magically by the solver
        to prevent infeasibility when the network is fundamentally starved.
        Rows = Time, Columns = Nodes.
        Default filename is results/diagnostics/time/generation.csv.
        """
        self._variable_table(filename, 'generation', 'generation', self._non_production_node_names())

    def diagnostic_absorption(self, filename: str = None) -> None:
        """
        Generates a CSV table detailing the ABSORPTION variable over time.
        Rows = Time, Columns = MATERIAL input nodes (Layer 0) which discard/waste physical oversource.
        Default filename is results/diagnostics/time/absorption.csv.
        """
        # Absorption realistically only occurs at MATERIAL nodes receiving constant external SUPPLY
        nodes_to_track = {n_str for n_str, t in self._network.nodes if self._network.nodes[(n_str, t)].get('type') == 'MATERIAL'}
        self._variable_table(filename, 'absorption', 'absorption', sorted(nodes_to_track, key=natural_sort_key))

    # helper methods
    def _layer_flow_table(self, filename: str, layer: int, storage_marker: str) -> None:
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"

        if not self._optimum_found:
            print(f"[WARNING] Optimization not found. '{filename}' will output 'na' for flows.")

        layer_char = ascii_uppercase[layer]

        # Collect relevant nodes: local storages of the requested kind and warehouses (W) belonging to the structural layer
        storage_nodes = []
        for n_str, n_data in self._baseNetwork.nodes(data=True):
            if n_str.startswith(layer_char):
                if storage_marker in n_str or '(W)' in n_str:
                    storage_nodes.append(n_str)

        # Group consecutively by production node naturally (e.g. A0, A1, A2... A10) and then by input naturally
        storage_nodes.sort(key=lambda x: (natural_sort_key(x.split('(')[0]), natural_sort_key(x)))

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time'] + storage_nodes)

            for t in range(self.duration):
                row = [t]
                for n_str in storage_nodes:
                    if self._optimum_found:
                        node_t = (n_str, t)
                        amount = 0.0
                        if node_t in self._network.nodes:
                            # 1. Sum incoming flows
                            for p in self._network.predecessors(node_t):
                                amount += self._network[p][node_t].get('flow', 0.0)
                            # 2. Add structural external generation/supply
                            amount += self._network.nodes[node_t].get('generation', 0.0)
                            amount += self._network.nodes[node_t].get('supply', 0.0)
                        row.append(f"{amount:.2f}")
                    else:
                        row.append("na")
                writer.writerow(row)

    def _non_production_node_names(self) -> list:
        nodes_to_track = set()
        for n_str, t in self._network.nodes:
            if '(PI)' not in n_str and '(PO)' not in n_str and n_str != 'AS':
                nodes_to_track.add(n_str)
        return sorted(nodes_to_track, key=natural_sort_key)

    def _variable_table(self, filename: str, attribute: str, description: str, nodes_to_track: list) -> None:
        if filename is None:
            filename = str(config.results_dir("diagnostics", "time") / f"{attribute}.csv")
        if not filename.endswith('.csv'): filename += '.csv'

        out_dir = os.path.dirname(filename)
        if out_dir: os.makedirs(out_dir, exist_ok=True)

        if not self._optimum_found:
            print(f"[WARNING] Optimization not found. '{filename}' will output 'na' for {description}.")

        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'] + nodes_to_track)
            for t in range(self.duration):
                row = [t]
                for n in nodes_to_track:
                    if self._optimum_found and (n, t) in self._network.nodes:
                        row.append(f"{self._network.nodes[(n, t)].get(attribute, 0.0):.2f}")
                    else:
                        row.append("na")
                writer.writerow(row)
