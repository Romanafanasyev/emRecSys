import pickle

db_path = './database.pkl'


with open(db_path, 'rb') as db_file:
    users_db = pickle.load(db_file)
print(users_db)

for user in users_db:
    print(user)
