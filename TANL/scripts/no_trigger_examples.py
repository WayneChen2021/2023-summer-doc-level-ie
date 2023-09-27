from process_og_muc import create_map, process_role
from nltk.tokenize import TreebankWordTokenizer as tbwt
import argparse
import json

def main(message_id_map, test_only=False):
    all_gtt = []
    all_tanl = []
    if test_only:
        sorted_keys = sorted(k for k in message_id_map.keys() if 'TST' in k)
    else:
        sorted_keys = sorted(k for k in message_id_map.keys())
    for k in sorted_keys:
        template_infos = message_id_map[k]
        text_as_str = template_infos['text'].replace("\n", " ")
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
        error_analysis_container_coref = {
            "docid": k,
            "tokens": tanl_template["tokens"],
            "templates": []
        }
        for template in template_infos["templates"]:
            if not "*" in template['INCIDENT: TYPE']:
                perp_ind, perp_ind_gtt = process_role(template, text_as_str, 'PERP: INDIVIDUAL ID')
                perp_org, perp_org_gtt = process_role(template, text_as_str, 'PERP: ORGANIZATION ID')
                target, target_gtt = process_role(template, text_as_str, 'PHYS TGT: ID')
                victim, victim_gtt = process_role(template, text_as_str, 'HUM TGT: NAME')
                weapon, weapon_gtt = process_role(template, text_as_str, 'INCIDENT: INSTRUMENT ID')

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
                    "PerpInd": [],
                    "PerpOrg": [],
                    "Target": [],
                    "Victim": [],
                    "Weapon": []
                }
                print(perp_ind, perp_org, target, victim, weapon)
                print(1/0)
        
        all_gtt.append(error_analysis_container)
    
    return all_tanl, all_gtt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--muc_dir", type=str, required=True)
    parser.add_argument("--tanl_out", type=str, required=False)
    parser.add_argument("--gtt_out", type=str, required=False)
    args = parser.parse_args()

    message_id_map = create_map(args.muc_dir)
    all_tanl, all_gtt = main(message_id_map, args.test_only)

    if args.tanl_out:
        with open(args.tanl_out, "w") as f:
            f.write(json.dumps(all_tanl))
    
    if args.gtt_out:
        with open(args.gtt_out, "w") as f:
            f.write(json.dumps(all_gtt))