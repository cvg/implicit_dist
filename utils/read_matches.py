import numpy as np

def read_matches(matches_path):
    matches = {}
    points2D = np.zeros((0,2))
    points3D = np.zeros((0,3))

    with open(matches_path) as f:
        lines = f.read().splitlines()

        for line in lines:
            row = line.split(' ')
            points2D = np.r_[ points2D, np.expand_dims(np.array(row[0:2],dtype=float),axis=0)]
            points3D = np.r_[ points3D, np.expand_dims(np.array(row[2:5],dtype=float),axis=0)]
            
    return points2D, points3D
        

def read_matches_list(dataset_path, max_queries=None):
    queries = []
    with open(dataset_path + '/image_list.txt') as f:
        lines = f.read().splitlines()
        for line in lines:
            row = line.split(' ')      
            query = {}  
            query['name'] = row[0]
            query['camera'] = {}
            query['camera']['model'] = row[1]
            query['camera']['width'] = int(row[2])
            query['camera']['height'] = int(row[3])
            n_params = len(row) - 11
            query['camera']['params'] = np.array(row[4:4+n_params],dtype=float) 

            query['qvec'] = np.array(row[4+n_params:8+n_params],dtype=float)
            query['tvec'] = np.array(row[8+n_params:],dtype=float)

            p2d, p3d = read_matches(dataset_path + '/' + query['name'] + '.matches.txt')
            query['matches'] = (p2d,p3d)

            queries.append(query)

            if max_queries is not None:
                if len(queries) >= max_queries:
                    break
    return queries
