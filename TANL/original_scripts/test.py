name_mapping = {
    "PerpInd": "perpetrating individual",
    "PerpOrg": "perpetrating organization",
    "Target": "target",
    "Victim": "victim",
    "Weapon": "weapon"
}
entities = []
relations = set()
templates = [{"PerpInd": [[[38, 39]]], "PerpOrg": [[[41, 46], [46, 48]]], "Target": [], "Victim": [], "Weapon": []}, {"PerpInd": [[[38, 39]]], "PerpOrg": [], "Target": [[[319, 322]]], "Victim": [], "Weapon": [[[348, 349]], [[241, 242]]]}]

for template in templates:
    simplified_template = {
        "PerpInd": [],
        "PerpOrg": [],
        "Target": [],
        "Victim": [],
        "Weapon": []
    }
    for role, entitiy_list in template.items():
        simplified_template[role] = [coref_set[0] for coref_set in entitiy_list]
    
    for entity_span_lists in simplified_template.values():
        entities += [span for span in entity_span_lists if not span in entities]

    for role1, entities1 in simplified_template.items():
        for role2, entities2 in simplified_template.items():
            if role2 != role1:
                for entity1 in entities1:
                    for entity2 in entities2:
                        if entity1 != entity2:
                            if entity1[0] > entity2[0]:
                                relations.add((
                                    "same event {} and {}".format(name_mapping[role2], name_mapping[role1]),
                                    entities.index(entity2),
                                    entities.index(entity1)
                                    ))
                            else:
                                relations.add((
                                    "same event {} and {}".format(name_mapping[role1], name_mapping[role2]),
                                    entities.index(entity1),
                                    entities.index(entity2)
                                    ))

print(entities)
print(relations)