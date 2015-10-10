import pickle
from sklearn import svm
import os
import arff

output_data ={ 0: 'angry', 1: 'happy', 2: 'neutral', 3:'unhappy'}

folder_path=os.path.dirname(os.path.realpath(__file__))
def read_test_data(filename):
    X_test= []
    file_name=filename.split(".")[0]
    input_file= folder_path+"/"+file_name+'.mp3'
    output_file=folder_path+"/"+file_name+'.wav'
    os.system("mpg123 -w "+output_file+ " "+ input_file+ " >/dev/null 2>&1")
    #print("mpg123 -w "+output_file+ " "+ input_file)
    
    file_with_path= output_file  
    os.system(folder_path+"/openEAR-0.1.0/SMILExtract -C "+folder_path+"/openEAR-0.1.0/config/emo_IS09.conf -I " + file_with_path + " -O "+folder_path+"/output.arff -instname inputN -classes numeric -classlabel 0" + " >/dev/null 2>&1")
    #print(file_with_path)

    data=arff.load(folder_path+ '/output.arff')
    data=list(data)
    
    x=[]
    for i in range(2,len(data[1])-1):
        x.append(data[1][i])
    X_test.append(x)
    os.system("rm "+output_file)
    os.system("rm "+folder_path+"/output.arff")
    return X_test
    
if __name__=="__main__":
    f_x= open('X_data.txt', 'rb')
    train_x_data=pickle.load(f_x)
    f_x.close
    f_y= open('Y_data.txt', 'rb')
    train_y_data=pickle.load(f_y)
    f_y.close
    f_fn= open('file_names.txt', 'rb')
    all_files=pickle.load(f_fn)
    f_fn.close

    #print("Training Started:")
    clf=svm.SVC()
    clf.fit(train_x_data, train_y_data)
    
    #print("prediction of training dataset ")
    #with open('training_output.csv','w') as outfile:
    #          outfile.write('file_name, outcome\n')
    #          for i in range(len(train_x_data)):
    #              output= clf.predict(train_x_data[i])
		  #print(output)
    #              outfile.write('%s,%s\n' %(all_files[i], (output_data[output[0]])))

    test_files=[]
    f=open('input.txt', 'r')
    for file in f:
        test_files.append(file)
    f.close()
    
    for i in range(len(test_files)):
        test_files[i]=test_files[i].rstrip()

    for i in test_files:
        test_x=read_test_data(i)
        output= clf.predict(test_x[0])
        print(output_data[output[0]])

    
