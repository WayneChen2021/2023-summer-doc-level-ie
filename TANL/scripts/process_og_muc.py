import json
import os
from nltk.tokenize import TreebankWordTokenizer as tbwt
import re
import argparse
from itertools import product
from functools import reduce

def span_mods(span):
    span = re.sub(r'\\', '', span)
    span = span.replace("  ", " ")
    span = span.replace("[ ", "[")
    return span

def process_role(template, text, role_name):
    output_lst = []
    entities = template[role_name].split("; ")
    for entity in entities:
        output_lst.append([span.strip()[1:-1] for span in re.split('[/?]', entity)])

    output_lst = [list(filter(lambda s: s.strip() not in ["-", "", "*"], sublist)) for sublist in output_lst]
    span_lst = [[span_mods(item) for item in sublist] for sublist in output_lst ]
    tup_lst = []
    gtt_lst = []
    for coref_spans in span_lst:
        coref_lst = []
        for span in coref_spans:
            head = text.find(span)
            try:
                assert head != -1
            except:
                print(span)
                raise Exception
            coref_lst.append((head, head + len(span), role_name))
        
        pos_sorted = sorted(coref_lst, key = lambda tup: tup[0])
        # len_sorted = sorted(coref_lst, key = lambda tup: tup[0] - tup[1])
        
        tup_lst += pos_sorted[:1]
        # tup_lst += len_sorted[:1]
        gtt_lst.append([text[tup[0] : tup[1]] for tup in coref_lst])
    
    return list(set(tup_lst)), gtt_lst

def build_entity(name, spans, head, tail):
    entity_head = -1
    entity_tail = -1
    for i, tup in enumerate(spans):
        if head >= tup[0] and head < tup[1]:
            entity_head = i
            break
    
    assert entity_head != -1
    for i, tup in enumerate(spans[(entity_head-1):]):
        if tail > tup[0] and tail <= tup[1]:
            entity_tail = entity_head + i
            break
    
    return {
        "type": name,
        "start": entity_head,
        "end": entity_tail
    }

def span_overlaps(span1, span2):
    earlier = span1
    later = span2
    if span2[0] < span1[0]:
        earlier = span2
        later = span1
    
    if later[0] < earlier[1]:
        return True
    
    return False

def create_map(original_muc_dir):
    message_id_map = {}
    for file in os.listdir(original_muc_dir):
        with open(os.path.join(original_muc_dir, file), "r") as f:
            og = json.loads(f.read())
            
            for example in og:
                message_id = example['template']['MESSAGE: ID'].strip()
                if not message_id in message_id_map:
                    message_id_map[message_id] = {'text': example['text'], 'templates': []}
                message_id_map[message_id]['templates'].append(example['template'])
    
    return message_id_map

def handle_edge_cases(matching_template, og_message_id):
    for k, v in matching_template.items():
        if 'ESPERANA' in v:
            matching_template[k] = v.replace('ESPERANA', 'ESPERANZA')
        if '"ALFREDO CRISTIANI"; "ROBERTO D\'AUBUISSON"' in v:
            matching_template[k] = v.replace('"ALFREDO CRISTIANI"; "ROBERTO D\'AUBUISSON"', '"ALFREDO] CRISTIANI"; "ROBERTO] D\'AUBUISSON"')
        if 'ALLEGED COMMANDOS OF THE \\"CINCHONEROS\\" PEOPLE\'S LIBERATION FRONT' in v:
            matching_template[k] = v.replace('ALLEGED COMMANDOS OF THE \\"CINCHONEROS\\" PEOPLE\'S LIBERATION FRONT', 'ALLEGED COMMANDOS OF THE \\"CINCHONERO\\" PEOPLE\'S LIBERATION FRONT')
        if '"CINCHONERO PEOPLE\'S LIBERATION FRONT"' == v:
            matching_template[k] = '"CINCHONERO" PEOPLE\'S LIBERATION FRONT'
        if 'ABUISSON' in v:
            matching_template[k] = v.replace('ABUISSON', 'AUBUISSON')
        if 'HONDUTEL OFFICE' in v:
            matching_template[k] = v.replace('HONDUTEL OFFICE', 'HONDUTEL [HONDURAN TELECOMMUNICATIONS ENTERPRISE] OFFICE')
        if 'WINDOWS OF A NEARBY BUILDING' in v:
            matching_template[k] = v.replace('WINDOWS OF A NEARBY BUILDING', 'WINDOWS OF NEARBY BUILDINGS')
        if og_message_id == 'DEV-MUC3-0315 (ITP, NYU)' and 'FARABUNDO MARTI NATIONAL LIBERATION FRONT' in v:
            matching_template[k] = v.replace('FARABUNDO MARTI NATIONAL LIBERATION FRONT', 'FARABUNDO MARTI NATIONAL LIBERATION MARTI FRONT')
        if og_message_id == 'DEV-MUC3-0319 (ITP, NYU)' and 'FARABUNDO MARTI NATIONAL LIBERATION MARTI FRONT' in v:
            matching_template[k] = v.replace('FARABUNDO MARTI NATIONAL LIBERATION MARTI FRONT', 'FARABUNDO MARTI NATIONAL LIBERATION FRONT')
        if 'DRUG TRAFFICKING CAPOS' in v:
            matching_template[k] = v.replace('DRUG TRAFFICKING CAPOS', 'DRUG TRAFFICKING "CAPOS"')
        if '? "TERRORIST" / "TERRORISTS"' == v:
            matching_template[k] = '"TERRORIST"'
        if '"ECOPETROL FUEL STORAGE TANKS"' in v:
            matching_template[k] = v.replace('"ECOPETROL FUEL STORAGE TANKS"', '"ECOPETROL [COLOMBIAN PETROLEUM ENTERPRISE] FUEL STORAGE TANKS"')
        if 'ATLACATL' in v:
            matching_template[k] = v.replace('ATLACATL', '"ATLACATL"')
        if 'A 7.65 MM AND A 9-MM' in v:
            matching_template[k] = v.replace('A 7.65 MM AND A 9-MM', 'A 7.65-MM AND THE OTHER A 9-MM')
        if og_message_id == 'DEV-MUC3-0442 (LSI)' and 'BERNARDETTE PARDO' in v:
            matching_template[k] = v.replace('BERNARDETTE PARDO', 'BERNARDETT PARDO')
        if 'CAMINO REAL HOTEL' in v:
            matching_template[k] = v.replace('CAMINO REAL HOTEL', '"CAMINO REAL" HOTEL')
        if "YPFB'S GUALBERTO VILLAROEL REFINERY" in v:
            matching_template[k] = v.replace("YPFB'S GUALBERTO VILLAROEL REFINERY", "YPFB'S [BOLIVIAN GOVERNMENT OIL DEPOSITS] GUALBERTO VILLAROEL REFINERY")
        if 'CONAVI BRANCH' in v:
            matching_template[k] = v.replace('CONAVI BRANCH', 'CONAVI [NATIONAL SAVINGS AND HOUSING CORPORATION] BRANCH')
        if 'COLDESARROLLO OFFICES' in v:
            matching_template[k] = v.replace('COLDESARROLLO OFFICES', 'COLDESARROLLO [EXPANSION UNKNOWN] OFFICES')
        if "\"UMOPAR [MOBILE UNITS FOR RURAL AREA] AGENTS\" / \"AGENTS\" " == v:
            matching_template[k] = "\"UMOPAR [MOBILE UNITS FOR RURAL AREAS] AGENTS\" / \"AGENTS\" "
        if "\"MAOIST-INCLINED LEFTIST ARMED GROUPING SENDERO LUMINOSO\" / \"SENDERO LUMINOSO\"" == v:
            matching_template[k] = '\"MAOIST-INCLINED LEFTIST ARMED GROUPING "SENDERO LUMINOSO."\" / \"SENDERO LUMINOSO\"'
        if "\"EXTREME RIGHTIST GROUPS\" / \"\\\"EXTREME RIGHTIST\\\" GROUPS\"; \"EXTREME LEFT\" / \"THE GUERRILLAS\" / \"GUERRILLAS\"" == v:
            matching_template[k] = '""EXTREME RIGHTIST" GROUPS"; \"EXTREME LEFT\" / \"THE GUERRILLAS\" / \"GUERRILLAS\"'
        if "\"EXTREME RIGHTIST GROUPS\" / \"\\\"EXTREME RIGHTIST\\\" GROUPS\"" == v:
            matching_template[k] = '""EXTREME RIGHTIST" GROUPS"'
        if "\"\"\"\"\"ATLACATL\"\"\"\" BATTALION\"" == v:
            matching_template[k] = '"ATLACATL" BATTALION'
        if "\"\\\"THE EXTRADITABLES\\\"\" / \"THE EXTRADITABLES\"" == v:
            matching_template[k] = '"THE EXTRADITABLES,"'
        if "\"\"\"ATLACATL\"\"\" BATTALION" == v:
            matching_template[k] = '"ATLACATL" BATTALION'
        if "\"\"ATLACATL\"\" BATTALION" == v:
            matching_template[k] = '"ATLACATL" BATTALION'
        if "\"SOVIET EMBASSY BUILDING\"; \"CAR\"; \"CARS\"" == v:
            matching_template[k] = "\"SOVIET EMBASSY BUILDING\"; \"CAR\"; \"VEHICLES\""
        if "\"FENSATRAS BUILDING\"" == v:
            matching_template[k] = 'FENSATRAS [SALVADORAN WORKERS NATIONAL UNION FEDERATION] BUILDING'
        if "\"CEL MINISTATION\" / \"LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION MINISTATION\"" == v:
            matching_template[k] = "\"CEL [LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION] MINISTATION\" / \"LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION MINISTATION\""
        if "\"CEL [LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION MINISTATION] MINISTATION\" / \"LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION MINISTATION\"" == v:
            matching_template[k] = "\"CEL [LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION] MINISTATION\" / \"LEMPA RIVER HYDROELECTRIC EXECUTIVE COMMISSION MINISTATION\""
        if "\"ALFRED CRISTIANI'S RIGHTIST GOVERNMENT\"; \"FARABUNDO MARTI NATIONAL LIBERATION FRONT ( FMLN)\" / \"FMLN\"" == v:
            matching_template[k] = "\"ALFREDO CRISTIANI'S RIGHTIST GOVERNMENT\"; \"FARABUNDO MARTI NATIONAL LIBERATION FRONT ( FMLN)\" / \"FMLN\"",
        if ["\"ALFREDO CRISTIANI'S RIGHTIST GOVERNMENT\"; \"FARABUNDO MARTI NATIONAL LIBERATION FRONT ( FMLN)\" / \"FMLN\""] == v:
            matching_template[k] = "\"ALFREDO CRISTIANI'S RIGHTIST GOVERNMENT\"; \"FARABUNDO MARTI NATIONAL LIBERATION FRONT (FMLN)\" / \"FMLN\""
        if ('"ALFREDO CRISTIANI\'S RIGHTIST GOVERNMENT"; "FARABUNDO MARTI NATIONAL LIBERATION FRONT ( FMLN)" / "FMLN"',) == v:
            matching_template[k] = '"ALFREDO CRISTIANI\'S RIGHTIST GOVERNMENT"; "FARABUNDO MARTI NATIONAL LIBERATION FRONT (FMLN)" / "FMLN"'
        if "\"DAS HEADQUARTERS\" / \"ADMINISTRATIVE DEPARTMENT OF SECURITY\" / \"DAS [ADMINISTRATIVE DEPARTMENT OF SECURITY] HEADQUARTERS\" / \"INTELLIGENCE HEADQUARTERS OF THE NATIONAL POLICE\" / \"GENERAL HEADQUARTERS OF THE COLOMBIAN INTELLIGENCE SERVICES\" / \"OFFICES OF THE DAS\"" == v:
            matching_template[k] = "\"ADMINISTRATIVE DEPARTMENT OF SECURITY\" / \"DAS [ADMINISTRATIVE DEPARTMENT OF SECURITY] HEADQUARTERS\" / \"INTELLIGENCE HEADQUARTERS OF THE NATIONAL POLICE\" / \"GENERAL HEADQUARTERS OF THE COLOMBIAN INTELLIGENCE SERVICES\" / \"OFFICES OF THE DAS\""
        if "\"LIEUTENANTS YUSHY RENE MENDOZA\" / \"YUSHY RENE MENDOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JOSE RICARDO ESPINOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"COL GUILLERMO ALFREDO BENAVIDES\" / \"ONE SALVADORAN COLONEL\" / \"SALVADORAN COLONEL\"; \"SUBLIEUTENANT GONZALO GUEVARA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SARGEANTS ANTONIO RAMIRO AVALOS\" / \"ANTONIO RAMIRO AVALOS\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"TOMAS ZARPATE CASILLO\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"CORPORAL ANGEL PEREZ VASQUEZ\" / \"ANGEL PEREZ VASQUEZ\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SOLDIERS OSCAR MARIANO AMAYA\" / \"OSCAR MARIANO AMAYA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JORGE ALBERTO SIERRA\" / \"FUGITIVE FROM JUSTICE\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"" == v:
            matching_template[k] = "\"LIEUTENANTS YUSHY RENE MENDOZA\" / \"YUSHY RENE MENDOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JOSE RICARDO ESPINOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"COL GUILLERMO ALFREDO BENAVIDES\" / \"ONE SALVADORAN COLONEL\" / \"SALVADORAN COLONEL\"; \"SUBLIEUTENANT GONZALO GUEVARA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SERGEANTS ANTONIO RAMIRO AVALOS\" / \"ANTONIO RAMIRO AVALOS\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"TOMAS ZARPATE CASTILLO\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"CORPORAL ANGEL PEREZ VASQUEZ\" / \"ANGEL PEREZ VASQUEZ\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SOLDIERS OSCAR MARIANO AMAYA\" / \"OSCAR MARIANO AMAYA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JORGE ALBERTO SIERRA\" / \"FUGITIVE FROM JUSTICE\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\""
        if "\"LIEUTENANTS YUSHY RENE MENDOZA\" / \"YUSHY RENE MENDOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JOSE RICARDO ESPINOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"COL GUILLERMO ALFREDO BENAVIDES\" / \"ONE SALVADORAN COLONEL\" / \"SALVADORAN COLONEL\"; \"SUBLIEUTENANT GONZALO GUEVARA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SARGEANT ANTONIO RAMIRO AVALOS\" / \"ANTONIO RAMIRO AVALOS\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"TOMAS ZARPATE CASILLO\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"CORPORAL ANGEL PEREZ VASQUEZ\" / \"ANGEL PEREZ VASQUEZ\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SOLDIERS OSCAR MARIANO AMAYA\" / \"OSCAR MARIANO AMAYA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JORGE ALBERTO SIERRA\" / \"FUGITIVE FROM JUSTICE\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"" == v:
            matching_template[k] = "\"LIEUTENANTS YUSHY RENE MENDOZA\" / \"YUSHY RENE MENDOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JOSE RICARDO ESPINOZA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"COL GUILLERMO ALFREDO BENAVIDES\" / \"ONE SALVADORAN COLONEL\" / \"SALVADORAN COLONEL\"; \"SUBLIEUTENANT GONZALO GUEVARA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SERGEANTS ANTONIO RAMIRO AVALOS\" / \"ANTONIO RAMIRO AVALOS\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"TOMAS ZARPATE CASTILLO\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"CORPORAL ANGEL PEREZ VASQUEZ\" / \"ANGEL PEREZ VASQUEZ\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"SOLDIERS OSCAR MARIANO AMAYA\" / \"OSCAR MARIANO AMAYA\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\"; \"JORGE ALBERTO SIERRA\" / \"FUGITIVE FROM JUSTICE\" / \"EIGHT OTHER MEMBERS OF THE SALVADORAN ARMY\" / \"MEMBERS OF THE SALVADORAN ARMY\""
        if "\"MEDELLIN DRUG CARTEL\" / \"DRUG CARTEL\"; \"EXTRADITABLES\" / \"THE \\\"EXTRADITABLES\\\"\" / \"\\\"EXTRADITABLES\\\"\" / \"THE EXTRADITABLES\"" == v:
            matching_template[k] = "\"MEDELLIN DRUG CARTEL\" / \"DRUG CARTEL\"; \"EXTRADITABLES\" / \"THE \\\"EXTRADITABLES\\\"\""
        if "\"DRUG \\\"BARONS\\\"\" / \"DRUG BARONS\"" == v:
            matching_template[k] = "\"DRUG \\\"BARONS\\\"\""
            
    return matching_template

def remove_dup_triggers(trigger_spans, incident_type):
    while True:
        has_overlap = False
        new_trigger_spans = set()
        for span in trigger_spans:
            overlapping = [s for s in trigger_spans if span_overlaps(s, span)]
            new_trigger_spans.add((min(overlapping, key=lambda x:x[0])[0], max(overlapping, key=lambda x:x[1])[1], incident_type, sum(s[-1] for s in overlapping)))
            if len(overlapping) > 1:
                has_overlap = True
        
        trigger_spans = new_trigger_spans
        if not has_overlap:
            break
    
    return trigger_spans

def sort_multi_trig(trigger_spans, k):
    pos_sorted = sorted(trigger_spans, key=lambda x:x[0])
    # popularity_sorted = sorted(trigger_spans, key=lambda x:x[-1])
    
    triggers = set()
    ind = 0
    while len(triggers) < k and len(triggers) < len(trigger_spans):
        triggers.add(pos_sorted[ind])
        # triggers.add(popularity_sorted[ind])
        ind += 1

    return list(triggers)

def enumerate_all_gold(entry, error_analysis_entry, name_lookup, triggers_per_temp):
    trigger_sets = list(product(*[range(len(sublist)) for sublist in entry['triggers']]))
    examples = []
    num_examples = 0
    if len(entry['triggers']):
        num_examples = reduce(lambda x , y : x * y, [min(len(sublist), triggers_per_temp) for sublist in entry['triggers']])
    # print(num_examples, len(trigger_sets))
    for trigger_set in trigger_sets:
        new_entry = {
            # "entities": [build_entity(name_lookup[tup[-1]], entry['token_spans'], tup[0], tup[1]) for tup in entry['entities']],
            "entities": [build_entity("template entity", entry['token_spans'], tup[0], tup[1]) for tup in entry['entities']],
            "triggers": [],
            "relations": [],
            "tokens": [entry['text'][tup[0]:tup[1]].lower().replace("[","(").replace("]",")") for tup in entry['token_spans']]
        }
        for template_ind, ind in enumerate(trigger_set):
            trigger_tup = entry['triggers'][template_ind][ind]
            # new_entry['triggers'].append(build_entity(trigger_tup[-2], entry['token_spans'], trigger_tup[0], trigger_tup[1]))
            # new_entry['triggers'].append(build_entity("trigger for {} event".format(trigger_tup[-2]), entry['token_spans'], trigger_tup[0], trigger_tup[1]))
            new_entry['triggers'].append(build_entity("trigger for {} event".format("{ " + trigger_tup[-2] + " }"), entry['token_spans'], trigger_tup[0], trigger_tup[1]))
            new_entry['relations'] += [{**existing, **{"tail": template_ind}} for existing in entry['relations'][template_ind]]
        if len(examples) != num_examples:
            if len(new_entry['triggers']) == len(set([str(e) for e in new_entry['triggers']])):
                examples.append(new_entry)
        else:
            break
    
    return examples, [error_analysis_entry] * len(examples)

def get_split(splits, ind, total_len):
    accum = 0
    for i, percent in enumerate(splits):
        accum += percent
        if ind + 1 <= total_len * accum:
            return i

def main(annotation_dir, message_id_map, triggers_per_temp=3, split_ind=None):
    error_analysis_templates = {}
    tanl_templates = {}
    name_lookup = {
        "PERP: INDIVIDUAL ID": "{ perpetrating individual }",
        "PERP: ORGANIZATION ID": "{ perpetrating organization }",
        "PHYS TGT: ID": "{ target }",
        "HUM TGT: NAME": "{ victim }",
        "INCIDENT: INSTRUMENT ID": "{ weapon }",
    }

    with open(annotation_dir, "r") as f:
        annotated_lst = json.loads(f.read())
    
    for annotation in annotated_lst:
        og_message_id = annotation['data']['template']['MESSAGE: ID'].strip()
        template_infos = message_id_map[og_message_id]
        message_id = og_message_id[:14]
        
        if message_id in tanl_templates:
            container = tanl_templates[message_id]
            error_analysis_container = error_analysis_templates[message_id]
        else:
            text_as_str = template_infos['text'].replace("\n", " ")
            container = {
                "text": text_as_str,
                "entities": [],
                "triggers": [], # list of lists of triggers (1 per possibility)
                "relations": [], # list of lists of relations (1 per possibility)
                "token_spans": list(tbwt().span_tokenize(text_as_str))
            }
            tanl_templates[message_id] = container
            error_analysis_container = {
                "docid": message_id,
                "doctext": text_as_str.lower(),
                "templates": []
            }
            error_analysis_templates[message_id] = error_analysis_container

        if len(template_infos['templates']) and not "*" in template_infos['templates'][0]['INCIDENT: TYPE']:
            matching_template = None
            for template in template_infos['templates']:
                if template['MESSAGE: TEMPLATE'].strip() == annotation['data']['template']['MESSAGE: TEMPLATE'].strip():
                    matching_template = template
                    break
        
            matching_template = handle_edge_cases(matching_template, og_message_id)

            incident_type = matching_template['INCIDENT: TYPE'].lower().strip()
            try:
                perp_ind, perp_ind_gtt = process_role(matching_template, container['text'], 'PERP: INDIVIDUAL ID')
                perp_org, perp_org_gtt = process_role(matching_template, container['text'], 'PERP: ORGANIZATION ID')
                target, target_gtt = process_role(matching_template, container['text'], 'PHYS TGT: ID')
                victim, victim_gtt = process_role(matching_template, container['text'], 'HUM TGT: NAME')
                weapon, weapon_gtt = process_role(matching_template, container['text'], 'INCIDENT: INSTRUMENT ID')
            except Exception as e:
                print(json.dumps(matching_template, indent=4))
                print(text_as_str)
                raise e
            trigger_spans = set()
            for trigger_annotation in annotation['annotations']:
                for single_annotation in trigger_annotation['result']:
                    if 'start' in single_annotation['value']:
                        trigger_spans.add((single_annotation['value']['start'], single_annotation['value']['end'], incident_type, 1))
            
            trigger_spans = remove_dup_triggers(trigger_spans, incident_type)
            multi_triggers = sort_multi_trig(trigger_spans, 10000)
            container['triggers'].append(multi_triggers)
            # container['triggers'].append(multi_triggers[:1])
            all_entities = list(set(perp_ind + perp_org + target + victim + weapon))
            # all_entities += list({(tup[0], tup[1], tup[2]) for tup in multi_triggers[1:]})
            for entity in all_entities:
                if not entity in container['entities']:
                    container['entities'].append(entity)

            gtt_template = {
                "incident_type": incident_type,
                "PerpInd": [[span.lower() for span in entity] for entity in perp_ind_gtt],
                "PerpOrg": [[span.lower() for span in entity] for entity in perp_org_gtt],
                "Target": [[span.lower() for span in entity] for entity in target_gtt],
                "Victim": [[span.lower() for span in entity] for entity in victim_gtt],
                "Weapon": [[span.lower() for span in entity] for entity in weapon_gtt]
            }
            for k, v in gtt_template.items():
                if k != "incident_type":
                    if v == [[]]:
                        gtt_template[k] = []
                    else:
                        for entity in v:
                            for tup in entity:
                                assert tup[1] != -1
            error_analysis_container['templates'].append(gtt_template)

            relation_set = []
            for entity_tup in all_entities:
                relation_set.append({
                        "type": "{} argument for ".format(name_lookup[entity_tup[-1]]) + "{ "  + incident_type + " } event",
                        # "type": "{}: {}".format(name_lookup[entity_tup[-1]], incident_type),
                        # "type": "{} argument for {} event".format(name_lookup[entity_tup[-1]], incident_type),
                        "head": container['entities'].index(entity_tup)
                    })
            container['relations'].append(relation_set)
    
    final_train_tanl_1, final_train_gtt_1, final_train_tanl_2, final_train_gtt_2 = [], [], [], []
    final_eval_tanl, final_eval_gtt = [], []
    for k in tanl_templates:
        split = 1
        if not 'TST' in k:
            split = 0
        entry = tanl_templates[k]
        if split != 0 and len(entry['triggers']) > 0:
            entry['triggers'] = [[trig[0]] for trig in entry['triggers']]
        tanls, gtts = enumerate_all_gold(entry, error_analysis_templates[k], name_lookup, triggers_per_temp)
        if len(tanls) == 0:
            tanls = [{
                "entities": [],
                "triggers": [],
                "relations": [],
                "tokens": [entry['text'][tup[0]:tup[1]].lower().replace("[","(").replace("]",")") for tup in entry['token_spans']],
            }]
            gtts = [error_analysis_templates[k]]
        
        for tanl in tanls:
            tanl["id"] = gtts[0]["docid"]

        if split == 0:
            if split_ind:
                if int(k[9:13]) < split_ind:
                    final_train_tanl_1 += tanls
                    final_train_gtt_1 += gtts
                else:
                    final_train_tanl_2 += tanls
                    final_train_gtt_2 += gtts
            else:
                if split == 0:
                    final_train_tanl_1 += tanls
                    final_train_gtt_1 += gtts
                elif split == 1:
                    final_eval_tanl += tanls
                    final_eval_gtt += gtts
        
        elif split == 1:
            final_eval_tanl += tanls
            final_eval_gtt += gtts
    
    return final_train_tanl_1, final_train_gtt_1, final_train_tanl_2, final_train_gtt_2, final_eval_tanl, final_eval_gtt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--muc_dir", type=str, required=True)
    parser.add_argument("--trigs_per_temp", type=int, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--tanl_train_out", type=str, required=False)
    parser.add_argument("--tanl_train_out_2", type=str, required=False)
    parser.add_argument("--tanl_eval_out", type=str, required=False)
    parser.add_argument("--gtt_train_out", type=str, required=False)
    parser.add_argument("--gtt_train_out_2", type=str, required=False)
    parser.add_argument("--gtt_eval_out", type=str, required=False)
    parser.add_argument("--split_ind", type=int, required=False)
    args = parser.parse_args()

    message_id_map = create_map(args.muc_dir)
    train_tanl_1, train_gtt_1, train_tanl_2, train_gtt_2, eval_tanl, eval_gtt = main(args.annotation_file, message_id_map, args.trigs_per_temp, args.split_ind)

    if args.tanl_train_out:
        with open(args.tanl_train_out, "w") as f:
            f.write(json.dumps(train_tanl_1))

    if args.tanl_eval_out:
        with open(args.tanl_eval_out, "w") as f:
            f.write(json.dumps(eval_tanl))
    
    if args.gtt_train_out:
        with open(args.gtt_train_out, "w") as f:
            f.write(json.dumps(train_gtt_1))

    if args.gtt_eval_out:
        with open(args.gtt_eval_out, "w") as f:
            f.write(json.dumps(eval_gtt))

    if args.tanl_train_out_2:
        with open(args.tanl_train_out_2, "w") as f:
            f.write(json.dumps(train_tanl_2))
    
    if args.gtt_train_out_2:
        with open(args.gtt_train_out_2, "w") as f:
            f.write(json.dumps(train_gtt_2))