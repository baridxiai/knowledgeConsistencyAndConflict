def reformat_prompt(template, subj_label, obj_label):
    output = template.replace('[X]', subj_label)
    output