def pick(filename): 
    import pickle
    pickle_off = open(filename,"rb")
    emp = pickle.load(pickle_off)
    print(emp)
