from process_og_muc import create_map, process_role, handle_edge_cases, build_entity
from nltk.tokenize import TreebankWordTokenizer as tbwt
from copy import deepcopy
import argparse
import json

def main(message_id_map, test_only=False):
    all_gtt = []
    all_tanl = []
    corefs = []
    if test_only:
        sorted_keys = sorted(k for k in message_id_map.keys() if 'TST' in k)
    else:
        sorted_keys = sorted(k for k in message_id_map.keys())
    for k in sorted_keys:
        template_infos = message_id_map[k]
        text_as_str = template_infos['text'].replace("\n", " ").replace("  ", " ")
        token_spans = list(tbwt().span_tokenize(text_as_str))
        tanl_template = {
            "entities": [],
            "triggers": [],
            "relations": [],
            "tokens": [text_as_str[tup[0]:tup[1]].lower().replace("[","(").replace("]",")") for tup in token_spans]
        }
        all_tanl.append(tanl_template)
        error_analysis_container = {
            "docid": k,
            "doctext": text_as_str.lower().replace("[","(").replace("]",")"),
            "templates": []
        }
        coref_container = {
            "docid": k,
            "tokens": tanl_template["tokens"],
            "templates": []
        }
        for template in template_infos["templates"]:
            if not "*" in template['INCIDENT: TYPE']:
                og_template = deepcopy(template)
                template = handle_edge_cases(template, k)
                perp_ind, perp_ind_gtt, perp_ind_coref = process_role(template, text_as_str, 'PERP: INDIVIDUAL ID', True)
                perp_org, perp_org_gtt, perp_org_coref = process_role(template, text_as_str, 'PERP: ORGANIZATION ID', True)
                target, target_gtt, target_coref = process_role(template, text_as_str, 'PHYS TGT: ID', True)
                victim, victim_gtt, victim_coref = process_role(template, text_as_str, 'HUM TGT: NAME', True)
                weapon, weapon_gtt, weapon_coref = process_role(template, text_as_str, 'INCIDENT: INSTRUMENT ID', True)

                token_sliced = [[], [], [], [], []]
                for i, outputs in enumerate([perp_ind_coref, perp_org_coref, target_coref, victim_coref, weapon_coref]):
                    for entity in outputs:
                        entity_info = []
                        for coref_tup in entity:
                            info = build_entity("", token_spans, coref_tup[0], coref_tup[1])
                            entity_info.append([info["start"], info["end"]])
                        token_sliced[i].append(entity_info)
                
                perp_ind_coref, perp_org_coref, target_coref, victim_coref, weapon_coref = token_sliced
                # print(perp_ind, perp_org)
                # print(1/0)

                gtt_template = {
                    "incident_type": template['INCIDENT: TYPE'].lower().strip(),
                    "PerpInd": [[span.lower().replace("[","(").replace("]",")") for span in entity] for entity in perp_ind_gtt if len(entity)],
                    "PerpOrg": [[span.lower().replace("[","(").replace("]",")") for span in entity] for entity in perp_org_gtt if len(entity)],
                    "Target": [[span.lower().replace("[","(").replace("]",")") for span in entity] for entity in target_gtt if len(entity)],
                    "Victim": [[span.lower().replace("[","(").replace("]",")") for span in entity] for entity in victim_gtt if len(entity)],
                    "Weapon": [[span.lower().replace("[","(").replace("]",")") for span in entity] for entity in weapon_gtt if len(entity)]
                }
                error_analysis_container["templates"].append(gtt_template)

                coref_template = {
                    "PerpInd": [[tup[:2] for tup in entity] for entity in perp_ind_coref if len(entity)],
                    "PerpOrg": [[tup[:2] for tup in entity] for entity in perp_org_coref if len(entity)],
                    "Target": [[tup[:2] for tup in entity] for entity in target_coref if len(entity)],
                    "Victim": [[tup[:2] for tup in entity] for entity in victim_coref if len(entity)],
                    "Weapon": [[tup[:2] for tup in entity] for entity in weapon_coref if len(entity)]
                }
                coref_container["templates"].append(coref_template)
        
        all_gtt.append(error_analysis_container)
        corefs.append(coref_container)
    
    return all_tanl, all_gtt, corefs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--muc_dir", type=str, required=True)
    parser.add_argument("--tanl_out", type=str, required=False)
    parser.add_argument("--gtt_out", type=str, required=False)
    parser.add_argument("--coref_out", type=str, required=False)
    args = parser.parse_args()

    message_id_map = create_map(args.muc_dir)
    all_tanl, all_gtt, corefs = main(message_id_map, args.test_only)

    if args.tanl_out:
        with open(args.tanl_out, "w") as f:
            f.write(json.dumps(all_tanl))
    
    if args.gtt_out:
        with open(args.gtt_out, "w") as f:
            f.write(json.dumps(all_gtt))
    
    if args.coref_out:
        with open(args.coref_out, "w") as f:
            f.write(json.dumps(corefs))