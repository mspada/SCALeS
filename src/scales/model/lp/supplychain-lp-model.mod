
param PENALTY >=0 default 1e9;

param DURATION, integer, >0; 
set TIME := 0..DURATION-1;

set BASIC_NODES;                                       
set NODES := {BASIC_NODES cross TIME};      

set ARCS within {NODES cross NODES};            
set IN_NEIGHBOURS{ NODES } within NODES;
set OUT_NEIGHBOURS{ NODES } within NODES;

param CAP{ ARCS } >= 0;              
param COST{ ARCS } >= 0;  
param SUPPLY{ NODES } >= 0 default 0;
param DEMAND{ NODES } >= 0 default 0;

param NUMBER_RULES_FOR{ NODES } default 1;
set RULES_FOR{ (n,t) in NODES } := 0..NUMBER_RULES_FOR[n,t]-1;
param IN_RULE_COEFFICIENTS{ (i,ti) in NODES, (j,tj) in IN_NEIGHBOURS[i,ti], RULES_FOR[i, ti] } default 0;
param OUT_RULE_COEFFICIENTS{ (i,ti) in NODES, (j,tj) in OUT_NEIGHBOURS[i,ti], RULES_FOR[i, ti] } default 0;

param BACKORDER_PENALTY >=0 default 1e6;
param ALLOW_BACKORDER{ BASIC_NODES } binary default 1;
param ALLOW_GENERATION{ BASIC_NODES } binary default 1;
param ALLOW_ABSORPTION{ BASIC_NODES } binary default 1;

var FLOW{ ARCS }, >= 0;
var GENERATION{ NODES }, >= 0;
var ABSORPTION{ NODES }, >= 0;
var BACKORDER{ NODES }, >= 0;

s.t. RESTRICT_BACKORDER{ (i, ti) in NODES : ALLOW_BACKORDER[i] == 0 }: BACKORDER[i, ti] = 0;
s.t. RESTRICT_GENERATION{ (i, ti) in NODES : ALLOW_GENERATION[i] == 0 }: GENERATION[i, ti] = 0;
s.t. RESTRICT_ABSORPTION{ (i, ti) in NODES : ALLOW_ABSORPTION[i] == 0 }: ABSORPTION[i, ti] = 0;

# FLOW is at most CAP on all edges.
s.t. FLOW_CAP_CONSTRAINT{ (i,ti,j,tj) in ARCS}: FLOW[ i,ti,j,tj ] <= CAP[ i,ti,j,tj ];
# FLOW is at least DEMAND_ARCS for all edges.
s.t. FLOW_RULE_CONSTRAINT{ (i,ti) in NODES, k in RULES_FOR[i,ti] }: 
    + sum{(j,tj) in IN_NEIGHBOURS[i,ti]} FLOW[ j,tj,i,ti ] * IN_RULE_COEFFICIENTS[i,ti,j,tj,k] 
    + SUPPLY[i,ti]
    + GENERATION[i, ti]
    - sum{(j,tj) in OUT_NEIGHBOURS[i,ti]} FLOW[ i,ti,j,tj ] * OUT_RULE_COEFFICIENTS[i,ti,j,tj,k]
    - DEMAND[i,ti]
    - ABSORPTION[i, ti]
    + BACKORDER[i, ti]
    - (if ti > 0 then BACKORDER[i, ti-1] else 0)
    =0; 

# Find a solution that minimises the total cost.
minimize obj: sum{ (i,ti,j,tj) in ARCS} COST[i,ti,j,tj] * FLOW[i,ti,j,tj] + sum{ (i, ti) in NODES} PENALTY * GENERATION[i, ti] + sum{ (i, ti) in NODES} BACKORDER_PENALTY * BACKORDER[i, ti];

solve; 


# printf "RESULTS\n";
printf { (i, ti) in NODES } "%10s %4d %4f %4f %4f\n", i, ti, GENERATION[i, ti], ABSORPTION[i, ti], BACKORDER[i, ti];
printf { (i, ti, j, tj) in ARCS }: "%10s %4d %10s %4d %4f\n", i, ti, j, tj, FLOW[i, ti, j, tj];
# printf "COST: %4f\n", sum{ (i,ti,j,tj) in ARCS} COST[i,ti,j,tj] * FLOW[i,ti,j,tj] + sum{ (i, ti) in NODES} PENALTY * GENERATION[i, ti];

end;
