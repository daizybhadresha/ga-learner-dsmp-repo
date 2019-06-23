# --------------
##File path for the file 
file_path 

def read_file(path):
    file = open(file_path,"r")
    sentence = file.read()
    file.close()
    return sentence

sample_message = read_file(file_path)
print(sample_message)
#Code starts here



# --------------
#Code starts here
def read_file(path):
    file = open(path,"r")
    sentence = file.read()
    file.close()
    return sentence

def fuse_msg(message_a, message_b):
    message_a = int(message_a)
    message_b = int(message_b)
    quotient = message_b//message_a
    return str(quotient)

message_1 = read_file(file_path_1)
message_2 = read_file(file_path_2)
print(message_1)
print(message_2)

secret_msg_1 = fuse_msg(message_1, message_2)
print(secret_msg_1)






# --------------
#Code starts here
def read_file(path):
    file = open(path,"r")
    sentence = file.read()
    file.close()
    return sentence

def substitute_msg(message_c):
    sub = ""
    if message_c == 'Red':
        sub = 'Army General'
    if message_c == 'Green':
        sub = 'Data Scientist'
    if message_c == 'Blue':
        sub = 'Marine Biologist'
    return sub

message_3 = read_file(file_path_3)
print(message_3)
secret_msg_2 = substitute_msg(message_3)
print(secret_msg_2)


# --------------
# File path for message 4  and message 5
file_path_4
file_path_5

def read_file(path):
    file = open(path,"r")
    sentence = file.read()
    file.close()
    return sentence

def compare_msg(message_d, message_e):
    a_list = message_d.split()
    b_list = message_e.split()
    c_list = [i for i in a_list if i not in b_list]
    final_msg = " ".join(c_list)
    return final_msg

message_4 = read_file(file_path_4)
message_5 = read_file(file_path_5)
print(message_4)
print(message_5)
secret_msg_3 = compare_msg(message_4,message_5)
print(secret_msg_3)

#Code starts here







# --------------
#Code starts here

def read_file(path):
    file = open(path,"r")
    sentence = file.read()
    file.close()
    return sentence

def extract_msg(message_f):
    a_list = message_f.split()
    even_word = lambda x : len(x)%2==0
    b_list = filter(even_word, a_list)
    final_msg = " ".join(b_list)
    return final_msg

message_6 = read_file(file_path_6)
print(message_6)
secret_msg_4 = extract_msg(message_6)
print(secret_msg_4)



# --------------
#Secret message parts in the correct order
message_parts=[secret_msg_3, secret_msg_1, secret_msg_4, secret_msg_2]


final_path= user_data_dir + '/secret_message.txt'

#Code starts here

def write_file(secret_msg, path):
    file = open(final_path,"a+")
    file.write(secret_msg)
    file.close()

secret_msg = " ".join(message_parts)
write_file(secret_msg, final_path)
print(secret_msg)




