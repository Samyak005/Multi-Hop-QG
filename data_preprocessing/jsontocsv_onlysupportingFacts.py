import json

file_input = "/home/samyak/Documents/IRE/hotpotqa/dataset/hotpot_train_v1.1.json"
file_output = "/home/samyak/Documents/IRE/hotpotqa/dataset/hotpot_train_fullcontext_v1.1.csv"

with open(file_input, "r", encoding='utf-8') as reader:
        data = json.load(reader)

csvfile = open(file_output, 'w')
csvfile.write(",question,text"+"\n")

count = 0
for line2 in data:
    context_list = line2['context']
    supporting_list = line2['supporting_facts']

#     print(len(context_list))

    supporting_title_list = []
    supporting_title_sentencenum = []
    line = line2
    line['context'] = ""

    for s in supporting_list:
        supporting_title_list.append(s[0])  # the supporting title

        for c in context_list:
            # print(c[0])
            # print(c[1])
            if s[0] == c[0]:
                if len(c[1]) > s[1]:
                    line['context'] += c[1][s[1]]               # the sentences
                else:
                    count += 1
                    # print("error", count)
        line['title'] = s[0] # the title
    line['label'] = 1
    # print(supporting_title_list, "-", line['context'])
    # print(line)
    l1 = line['question'].replace(',', '') 
    l1 = l1.replace('"', '')
    l1 = l1.replace(';', '')
    l1 = l1.replace(':', '')

    l2 =  "<answer> " + line['answer'].replace(',', '') + " <context> " + line['context'].replace(',', '') 

    l2 = l2.replace('"', '')
    l2 = l2.replace(';', '')
    l2 = l2.replace(':', '')

    line1 = line['_id'] + "," + l1+ "," + '"'+ l2 + '"'

    # print(line1)
    csvfile.write(line1+"\n")
    

csvfile.close()