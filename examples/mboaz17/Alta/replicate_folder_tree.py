inputpath = '/media/isl12/Alta/V7_Exp_25_1_21'
outputpath = '/media/isl12/Alta/V7_Exp_25_1_21_annot'

for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath, os.path.relpath(dirpath, inputpath))
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")
