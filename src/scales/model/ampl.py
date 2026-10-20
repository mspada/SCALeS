"""
Export of the time-expanded network as AMPL data (.dat) for the LP model in lp/supplychain-lp-model.mod.
"""

from itertools import product

AMPL_TEMPLATE = \
"""
param DURATION := {};

set BASIC_NODES := {} ;

param SUPPLY:=
{}
;

param DEMAND:=
{}
;

param:
    ARCS: COST CAP:=
{}
;

{}

{}

param NUMBER_RULES_FOR :=
{}
;

param IN_RULE_COEFFICIENTS :=
{}
;

param OUT_RULE_COEFFICIENTS :=
{}
;

param ALLOW_BACKORDER :=
{}
;

param ALLOW_GENERATION :=
{}
;

param ALLOW_ABSORPTION :=
{}
;

end;
"""


class AmplExportMixin:
    """Methods of SupplyChain that serialise `self._network` into AMPL data."""

    def to_ampl_data(self) -> str:
        """Returns the AMPL data (.dat) describing the dynamic network, as consumed by the LP model."""
        return AMPL_TEMPLATE.format(
                self._toAMPLDuration(),
                self._toAMPLBasicNodes(),
                self._toAMPLSupply(),
                self._toAMPLDemand(),
                self._toAMPLArcsCostCap(),
                self._toAMPLInNeighbours(),
                self._toAMPLOutNeighbours(),
                self._toAMPLNumberRulesFor(),
                self._toAMPLInRuleCoefficients(),
                self._toAMPLOutRuleCoefficients(),
                self._toAMPLAllowBackorder(),
                self._toAMPLAllowGeneration(),
                self._toAMPLAllowAbsorption()
        )

    def _toAMPLDuration(self) -> str:
        return self.duration

    def _toAMPLBasicNodes(self) -> str:
        return "'" + "' '".join(self.nodes) + "'"

    def _toAMPLSupply(self) -> str:
        amplSupply = ""
        for i, ti in self._network.nodes:
            supply = self._network.nodes[(i,ti)]['supply']
            amplSupply += f"'{i}', {ti} {supply} \n"

        return amplSupply

    def _toAMPLDemand(self) -> str:
        amplDemand = ""

        for i, ti in self._network.nodes:
            demand = self._network.nodes[(i,ti)]['demand']
            amplDemand += f"'{i}', {ti} {demand} \n"

        return amplDemand

    def _toAMPLArcsCostCap(self) -> str:
        amplArcsCostCap = ""

        for (i, ti), (j, tj) in self._network.edges:
            cost, cap = self._network[(i, ti)][(j, tj)]['cost'], self._network[(i, ti)][(j, tj)]['cap']
            amplArcsCostCap += f"    '{i}', {ti}, '{j}', {tj} {cost} {cap} \n"

        return amplArcsCostCap

    def _toAMPLInNeighbours(self) -> str:
        amplInNeighbours = ""

        for i, ti in self._network.nodes:
            amplInNeighbours += f"set IN_NEIGHBOURS['{i}', {ti}] := "
            for j, tj in self._network.predecessors((i, ti)):
                amplInNeighbours += f"('{j}', {tj}) "

            amplInNeighbours += "; \n"

        return amplInNeighbours

    def _toAMPLOutNeighbours(self) -> str:
        amplOutNeighbours = ""

        for i, ti in self._network.nodes:
            amplOutNeighbours += f"set OUT_NEIGHBOURS['{i}', {ti}] := "
            for j, tj in self._network.successors((i, ti)):
                amplOutNeighbours += f"('{j}', {tj}) "

            amplOutNeighbours += "; \n"

        return amplOutNeighbours

    def _toAMPLNumberRulesFor(self) -> str:
        amplRules = ""
        for i, ti in self._network.nodes:
            nRules = len(self._network.nodes[(i, ti)]['rule']) if self._canProduce(i, ti) else 1
            amplRules += f"'{i}', {ti} {nRules} \n"
        return amplRules

    def _toAMPLInRuleCoefficients(self) -> str:
        amplInRules = ""

        self._checkProductionCapabilities()

        for i, ti in self._network.nodes:
            if self._network.nodes[(i,ti)]['type'] in ('PRODUCTION', 'PRODUCTION_INPUT') and self._network.nodes[(i,ti)].get('prod_pred'):
                for index, j in enumerate(self._network.nodes[(i,ti)]['prod_pred']):
                    for k, tk in self._network.nodes[(i, ti)]['supp_pred']:
                        amplInRules += f"'{i}' {ti}, '{k}', {tk}, {index} 1 \n "
                    for tj in self._network.nodes[(i,ti)]['prod_pred'][j]:
                        amplInRules += f"'{i}', {ti}, '{j}', {tj}, {index} {1/self._network.nodes[(i, ti)]['rule'][j]} \n"
            elif self._network.nodes[(i,ti)]['type'] in ('PRODUCTION', 'PRODUCTION_INPUT'):
                for j, tj in self._network.nodes[(i, ti)]['supp_pred']:
                    amplInRules += f"'{i}' {ti}, '{j}', {tj}, 0 1 \n "
                for j, tj in sorted(set(self._network.predecessors((i,ti))) - set(self._network.nodes[(i,ti)]['supp_pred'])):
                    amplInRules += f"'{i}' {ti}, '{j}', {tj}, 0 0 \n "
            else:
                for j, tj in self._network.predecessors((i,ti)):
                    amplInRules += f"'{i}', {ti}, '{j}', {tj}, 0 1 \n"

        return amplInRules

    def _toAMPLOutRuleCoefficients(self) -> str:
        amplOutRules = ""

        for i, ti in self._network.nodes:
            nRules = len(self._network.nodes[(i, ti)]['rule']) if self._canProduce(i, ti) else 1
            for (j, tj), index in product(self._network.successors((i,ti)), range(nRules)):
                amplOutRules += f"'{i}', {ti}, '{j}', {tj}, {index} 1 \n"

        return amplOutRules

    # the basic node names are sorted so that the generated data is identical between runs
    def _basic_node_names(self) -> list:
        return sorted(set(i for (i,t) in self._network.nodes))

    def _toAMPLAllowBackorder(self) -> str:
        amplAllow = ""
        for n in self._basic_node_names():
            if '(PI)' in n or '(PO)' in n or n == 'AS':
                amplAllow += f"'{n}' 0\n"
            else:
                amplAllow += f"'{n}' 1\n"
        return amplAllow

    def _toAMPLAllowGeneration(self) -> str:
        amplGen = ""
        for n in self._basic_node_names():
            # Only allow generation on local storages (I) and warehouses (W)
            if '(I' in n or '(W)' in n:
                amplGen += f"'{n}' 1\n"
            else:
                amplGen += f"'{n}' 0\n"
        return amplGen

    def _toAMPLAllowAbsorption(self) -> str:
        amplAbs = ""
        for n in self._basic_node_names():
            # Only allow absorption on MATERIAL input nodes to force downstream backpressure
            if self._network.nodes[(n, 0)].get('type') == 'MATERIAL':
                amplAbs += f"'{n}' 1\n"
            else:
                amplAbs += f"'{n}' 0\n"
        return amplAbs
