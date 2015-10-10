from sklearn import svm
import os
import subprocess
import arff
import pickle
folder_path=os.path.dirname(os.path.realpath(__file__))
angry_folder=os.path.dirname(os.path.realpath(__file__))+ '/training_dataset/angry/'
happy_folder=os.path.dirname(os.path.realpath(__file__))+ '/training_dataset/happy/'
neutral_folder=os.path.dirname(os.path.realpath(__file__))+'/training_dataset/neutral/'
unhappy_folder=os.path.dirname(os.path.realpath(__file__))+'/training_dataset/unhappy/'

test_folder= '.'

folder_list=[angry_folder,happy_folder,neutral_folder,unhappy_folder] 

emotion = {'angry':0,'happy':1,'neutral':2,'unhappy':3}

def read_train_data(folder):
    X=[]
    Y=[]
    files=[]
    y=0
    if 'angry' in folder:
        y=0
        folder_name="./angry/"
    if 'happy' in folder:
        y=1
        folder_name="./happy/"
    if 'neutral' in folder:
        y=2
        folder_name="./neutral/"
    if 'unhappy' in folder:
        y=3
        folder_name="./unhappy/"
    
    for file in os.listdir(folder):
        file_with_path=folder+file
        
        files.append(folder_name+file)
        print(folder_path+"/openEAR-0.1.0/SMILExtract -C "+folder_path+"/openEAR-0.1.0/config/emo_IS09.conf -I " + file_with_path + " -O "+folder_path+"/output.arff -instname inputN -classes numeric -classlabel 0")
        os.system(folder_path+"/openEAR-0.1.0/SMILExtract -C "+folder_path+"/openEAR-0.1.0/config/emo_IS09.conf -I " + file_with_path + " -O "+folder_path+"/output.arff -instname inputN -classes numeric -classlabel 0 >/dev/null 2>&1" )

        data = arff.load(folder_path+ "/output.arff") 
        data=list(data)
        x=[]
        for i in  range(2,len(data[1])-1):
            x.append(data[1][i])

        X.append(x)
        Y.append(y)
        os.system("rm "+folder_path+ "/output.arff") 
    return files, X, Y    
def read_test_data(test_folder):
    print("hello")



    
def all_folders(folder_list):
    for i in folder_list:
        read_data(i)


if __name__=="__main__":
    
    file1, X1,Y1 = read_train_data(angry_folder)
    print(len(X1),len(Y1))
    
    file2,X2,Y2 = read_train_data(happy_folder)
    print(len(X2),len(Y2))
    
    file3,X3,Y3 = read_train_data(neutral_folder)
    print(len(X3),len(Y3))
    
    file4, X4,Y4 = read_train_data(unhappy_folder)
    print(len(X4),len(Y4))
    
    X_train= X1+X2+X3+X4
    Y_train= Y1+Y2+Y3+Y4
    all_files = file1+file2+file3+file4

    print(len(X_train), len(Y_train), len(all_files))

    fn_x=open('X_data.txt', 'wb')
    fn_y=open('Y_data.txt', 'wb')
    fn_files=open('file_names.txt','wb')
    pickle.dump(X_train,fn_x)
    pickle.dump(Y_train, fn_y)
    pickle.dump(all_files, fn_files)
    fn_x.close()
    fn_y.close()
    fn_files.close()

    
    print("training Started")
    clf=svm.SVC()
    clf.fit(X_train, Y_train)
    print("writing prediction of training dataset")

    with open('training_output.csv','w') as outfile:
              outfile.write('file_name, outcome\n')
              for i in range(len(X_train)):
                  output= clf.predict(X_train[i])
                  outfile.write('%s,%s\n' %(all_files[i], str(output)))

    
    
