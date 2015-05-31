# coding: utf8
 
'''
Created on 6 août 2014

@author: qgthurier
'''

import DataProcessingFactory as dpf
import unittest, numpy


class TestDataProcessingFactory(unittest.TestCase):
        
    def setUp(self):
        self.factory = dpf.DataProcessingFactory(training_file = 'utest.train.data', 
                                            test_file = 'utest.test.data',
                                            target_label = 'salary-class',
                                            verbose=False)
        self.factory.load_data()
        
    def tearDown(self):
        pass

    def test_nb_rows_train(self):
        # check that the number of lines imported is correct
        self.assertEqual(self.factory.train_set.shape[0], 21705)
        
    def test_nb_cols_train(self):
        # check that the number of column imported is correct
        self.assertEqual(self.factory.train_set.shape[1], 15)

    def test_nb_rows_test(self):
        # check that the number of lines imported is correct
        self.assertEqual(self.factory.test_set.shape[0], 10854)
        
    def test_nb_cols_test(self):
        # check that the number of column is correct
        self.assertEqual(self.factory.test_set.shape[1], 14)
        
    def test_label_encoding(self):
        # check that factors from train and test sets are encoded with the same integers
        # 2nd line in train set and 8th line in test set have same values for factors workclass & education
        self.assertTrue(self.factory.train_set.iloc[1]['education'] == self.factory.test_set.iloc[7]['education'])
        self.assertTrue(self.factory.train_set.iloc[1]['workclass'] == self.factory.test_set.iloc[7]['workclass'])
     
    def test_one_hot_encoding(self):
        self.factory.excluded_features = []
        self.factory.excluded_features += ['age','workclass','fnlwgt','education',
                                             'education-num','marital-status',
                                             'occupation','relationship','race',
                                             'capital-gain','capital-loss,hours-per-week',
                                             'native-country'] # we encode only the factor sex
                             
        self.factory.encode_data()
        # check that the number of binary features added after one-hot encoding is correct
        # since the factor sex has only two unique values, one-hot encoding increases the number of column by 1 
        self.assertEqual(self.factory.test_set.shape[1], 14 + 1)
        self.assertEqual(self.factory.train_set.shape[1], 15 + 1)
        # check that new columns added after one hot encoding corresponds between train and test sets
        # 1st record in train and test files are female
        self.assertTrue(self.factory.train_set.iloc[0]['sex_0'] == self.factory.test_set.iloc[0]['sex_0'])
        self.assertTrue(self.factory.train_set.iloc[0]['sex_1'] == self.factory.test_set.iloc[0]['sex_1'])
        
    def test_scaling(self):
        self.factory.excluded_features = []
        self.factory.excluded_features += ['workclass','fnlwgt','education',
                                            'education-num','marital-status',
                                            'occupation','relationship','race',
                                            'capital-gain','capital-loss,hours-per-week',
                                            'native-country', 'sex'] # we scale only the age
        self.factory.scale_data()
        # check that the sacling range is ok                     
        self.assertTrue(self.factory.train_set['age'].max() == 1.0)
        self.assertTrue(self.factory.train_set['age'].min() == 0.0)
        self.assertTrue(self.factory.test_set['age'].max() <= 1.0)
        self.assertTrue(self.factory.test_set['age'].min() >= 0.0)
        # check that the scaling keep equality between train and test sets
        # 3rd record in train and test files are 21 years old
        self.assertTrue(self.factory.train_set.iloc[2]['age'] == self.factory.test_set.iloc[3]['age'])
        
    def test_median_imputation(self):
        self.factory.excluded_features = []
        self.factory.excluded_features += ['age', 'education',
                                            'education-num','marital-status',
                                            'occupation','relationship','race',
                                            'capital-gain','capital-loss,hours-per-week',
                                            'native-country', 'sex'] # we impute only fnlwgt and workclass
        self.factory.impute_data()
        self.assertEqual(self.factory.test_set.iloc[1]['fnlwgt'], self.factory.train_set['fnlwgt'].median())
        self.assertEqual(self.factory.train_set.iloc[2]['fnlwgt'], self.factory.train_set['fnlwgt'].median())
        
    def test_mode_imputation(self):
        self.assertEqual(self.factory.test_set.iloc[2]['workclass'], self.factory.train_set['workclass'].mode()[0])
        self.assertEqual(self.factory.train_set.iloc[98]['workclass'], self.factory.train_set['workclass'].mode()[0])
                         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()