CREATE (manager:Person:Customer:Employee {property_merda: 'Alex Johnson'}),
       (assistant:Employee {property_merda: 'Taylor Smith'}),
       (manager)-[:MANAGES {since: date()}]->(assistant)
RETURN manager, assistant