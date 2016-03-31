import graphlab as gl
from graphlab import model_parameter_search
import pdb
from datetime import datetime
import time

def main():
    #read in customer_static. This is our original dataset
    cust_orig  = gl.SFrame.read_csv('train.csv')

    
      # return x == '1800-01-02'


    
    #Create the target column
    cust_orig['Label'] = cust_orig['ACTION']

    #add in military status classifier
    # mil_status = gl.SFrame.read_csv('data/new_mil_status_over_time.csv')

    # r = mil_status[cols].apply(lambda x: 'R' in x.values())

    # cust_orig.add_column(r, name='R')


    #Remove columns we think are not useful and remove labels
    cust_orig.remove_columns(['ACTION'])

    # cust_orig.remove_columns([ 'USAA_PARTY_SK'
    #                          , 'CUST_RESCIND_DT'
    #                          , 'MIL_BRCH_SRVC_CHANGE_FLAG'])

    train, test = cust_orig.random_split(0.5)

    print(cust_orig.column_names())

    model  = gl.logistic_classifier.create(train, 'Label')
    # model = gl.boosted_trees_classifier.create(train, target='Label')
    # model = gl.boosted_trees_classifier.create(train, target='Label', class_weights='auto')
    #model = gl.boosted_trees_classifier.create(train, target='Label')


    predictions = model.classify(test)
    results = model.evaluate(test)

    print(results)


    #Make actual model, get actual results.
    #read in provided test data
    test_data  = gl.SFrame.read_csv('test.csv')
    model  = gl.logistic_classifier.create(cust_orig, 'Label')
    #model = gl.boosted_trees_classifier.create(cust_orig, target='Label')
    predictions = model.classify(test_data)
    print(type(predictions))

    answer = gl.SFrame(data=test_data['id'])
    answer.rename({'X1':'Id'})
    answer.add_column(predictions['class'],name = 'Action')
    answer.save('answer.csv', format='csv')

main()
