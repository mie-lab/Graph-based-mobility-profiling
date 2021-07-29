
### current data format 

dictionary with user id as keys and _activity graph objects_ as values e.g.,

```python
{1597: <future_trackintel.activity_graph.activity_graph at 0x158eee23520>,
1598: <future_trackintel.activity_graph.activity_graph at 0x158eee23fa0>,
1599: <future_trackintel.activity_graph.activity_graph at 0x158f0883c70>}
```

The activity graph class is defined in `future_trackintel.activity_graph`

#### attributes 
- G: the nx graph model
    - node ids start with 0
    - nodes have a location id, geometry and an optional extent geometry
    - It is planned that arbitrary features of staypoints, locations and context will be
    assigned to nodes
    - edges are directed 
    - can be adressed by the 3-tuple (node 1, node 2, edge type) e.g., 
      `G.edges[(0,2, 'transition_counts')]`. _transition_counts_ is currently the only edge type that is used at the moment.