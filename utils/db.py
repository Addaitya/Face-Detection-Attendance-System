from pymongo import MongoClient

class PersonCollection:
    def __init__(self, uri:str, *, db_name:str='Face', collection_name:str='Embeddings'):
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

        except Exception as e:
            print("Error connecting to database server: \n{e}")

    def add_person(self, person:dict):
        keys = ['name', 'student_id', 'embedding']
        if not all(key in person for key in keys):
            raise ValueError("person must contain name, student_id and embedding field")
        
        person = {key:person[key] for key in keys}
        return self.collection.insert_one(person)

    def check_person(self, student_id: str)->bool:
        cnt = self.collection.count_documents({"student_id": student_id})
        if cnt:
            return True
        return False

    def search(self, embedding:list[int], index_name:str, field: str):
        '''
        Do vector search to find most relevant face

        embedding list(int): The encoded image of size 1024
        index_name str: name to the index to use of searching
        '''
        query = {
            "$vectorSearch": {
                "index": index_name,
                "limit": 4,
                "numCandidates": 100,
                "path": field,
                "queryVector": embedding,
            }
        }

        get_fields = {
            "$project": {
                '_id' : 1,
                'name' : 1,
                'student_id' : 1,
                "search_score": { "$meta": "vectorSearchScore" }
            }
        }
        try:
            result = self.collection.aggregate([
                query, 
                get_fields
            ])
            return list(result)
        except Exception as e:
            print(f'Error in searching: \n{e}')