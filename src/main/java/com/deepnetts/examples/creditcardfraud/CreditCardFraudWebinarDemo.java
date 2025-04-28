package com.deepnetts.examples.creditcardfraud;

import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

/**
 * Credit Card Fraud Detection. 
 * This example demonstrates how to use Feed Forward Neural Network to perform binary classification task.
 * The trained network classifies credit card transactions into one of two possible categories: fraud / not fraud (true/false)
 * The output of the network is probability that a given transaction is fraud.
 *
 * Data Set description.
 * The data set contains transactions made by credit cards in September 2013 by European cardholders.
 * This data set presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
 * The data set is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
 * For the purposes of this example data set has been balanced by using a subset of the original data set.
 * All attributes in data set are  anonymized except last two which represent transaction amount and class. 
 * URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download-latest
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart/
 * 
 */
public class CreditCardFraudWebinarDemo {

    public static void main(String[] args) throws IOException {
                       
        // specify CSV file options
        CsvReadOptions.Builder builder = 
	CsvReadOptions.builder("datasets/creditcard.csv") // csv file path
		.separator(',') // values are coma-delimited
		.header(true); // first line contains column names
        CsvReadOptions options = builder.build();

        // load data into a data frame
        Table dataTable = Table.read().usingOptions(options);
        
        // Prepare data for training
        DataPreparation dataPrep = new DataPreparation(dataTable);

        // print first few rows to see what's loaded
        dataPrep.previewRows(5);
                        
        // print columns with corresponding types
        dataPrep.columnInfo();        
              
        // remove column Time since it is not relevant
        System.out.println("Removing column Time"); 
        dataTable.removeColumns("Time");

        // remove duplicate rows if there are any
        dataTable.dropDuplicateRows();
        
        // check if there are any missing values
        dataPrep.countMissingValues();
        dataPrep.handleMissingValues();
                        
        // print basic statistics summary
        dataPrep.statistics();

        // check if data set is balanced - is there the same amount of positive or negative examples
        dataPrep.checkClassBalance("Class");
        
        // create balanced subset
        Table balancedData  = DataPreparation.createBalancedSample(dataTable, "Class", 1);
        dataPrep.checkClassBalance("Class");
                                   
        // create data set for neural network training   
        TabularDataSet dataSet = DataPreparation.createDataSet(balancedData );
        DataSets.scaleToMax(dataSet);
        
        // split data into training and test set
        TrainTestSplit split = DataSets.trainTestSplit(dataSet, 0.6);
        DataSet<MLDataItem> trainingSet = split.getTrainingSet();
        DataSet<MLDataItem> testSet = split.getTestSet();
        
        int numInputs= 29;
        int numOutputs = 1;
        
        // create instance of feed forward neural network using its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)   // size of the input layer corresponds to number of inputs
                .addFullyConnectedLayer(32,ActivationType.TANH) 
                .addOutputLayer(numOutputs, ActivationType.SIGMOID) // size of output layer corresponds to number of outputs, which is 1 for binary classification problems, and sigmoid transfer function is used for binary classification
                .lossFunction(LossType.CROSS_ENTROPY) // cross entropy loss function is commonly used for classification problems
                .build();
  
        // set parameters of the training algorithm
        neuralNet.getTrainer().setStopError(0.02f)  // training will stop when this error is reached
                              .setStopEpochs(10000) // or this number of epochs is reached (training iterations)
                              .setLearningRate(0.01f); // controls size of learning step ~ 1% of error
     
        neuralNet.train(trainingSet);
        
        // test neural network and print evaluation metrics
        EvaluationMetrics em = neuralNet.test(testSet);
        System.out.println(em);
        System.out.println("Done!");        
    }
}

