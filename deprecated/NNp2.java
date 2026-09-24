package deprecated;
import java.io.*;
import java.util.*;


/************************************** 
 * Garrett Williams                   *
 * 101-67-111                         *
 * CSC-475                            *
 * Assigment 2: part 2                *
 * ************************************
 * Handwitten digit classifier        *
 *                                    *
 * Fully-connected feedforward neural *
 * network utilizing backpropagation  *
 * and stochastic gradient descent to *
 * learn and classify digits from the *
 * public MNIST training and testing  *
 * data sets.                         *
 * ************************************
 */

public class NNp2{
  /*
   ******************
   *      MAIN      *
   ******************
  */
  public static void main(String[] args){
    // init network
    createNetwork();
    // call user interface
    startMainInterface();
  }

  /*
   ******************
   * Neural Network *
   ******************
  */
  // Debug/testing constants
  static final boolean DEBUG = false;  // prints additional network info during training/tests for debugging
  static final boolean TESTCHECK = false;  // uses the hard coded weights and biases for part 1
  static final boolean DIGIT_TRACK = true;  // track the accuracy stats of individual digits
  static final boolean SHOW_DIST = false;  // show distribution for output when running network on testing data

  // Network specifications constants
  static final int HIDDEN_LAYERS = 15;  // number of hidden layers to use in network
  static final int INPUT_SIZE = 784;  // number of nodes in input layer
  static final int HIDDEN_SIZE = 75;  // number of nodes in hidden layers
  static final int OUTPUT_SIZE = 10;  // number of nodes in output layer

  static final int BATCH_SIZE = 10;  // batch size used for network training
  static final int EPOCHS = 10;  // number of training epochs to perform          
  static final double ETA = 0.15f;  // learning rate 'eta'

  // Storage for weights and biases
  static double[][] bias = new double[HIDDEN_LAYERS][HIDDEN_SIZE];  // stores biases for all nodes in all hidden layers
  static double[] bias_out = new double[OUTPUT_SIZE];  // stores biases for all nodes in output layer

  static double[][] weights_in = new double[HIDDEN_SIZE][INPUT_SIZE];  // weights: input --> hidden layer
  static double[][][] weights = new double[Math.max(0, HIDDEN_LAYERS-1)][HIDDEN_SIZE][HIDDEN_SIZE];  // all hidden layer --> hidden layer weights
  static double[][] weights_out = new double[OUTPUT_SIZE][HIDDEN_SIZE];  // weights: hidden --> output layer
  
  // Storage for activations in each hidden layer
  static double[][] activs_hid;

  // Storage for weight gradients and errors
  static double[][] wG1 = new double[3][4];  // for printing weight gradients during testcheck (for part 1)
  static double[][] wG2 = new double[2][3];  // ... 
  
  static double[] grad_bias_out, err_out;  // stores bias gradient and errors for neurons in final layer (respectively)
  static double[][] err_hid, grad_bias_hid, grad_wts_in, grad_wts_out;  // stores errors and bias grads for each neuron in each hidden layer, weight gradient for each neuron from input and to output
  static double[][][] grad_wts_hid;  // stores weight gradients for each neuron in each hidden --> hidden layer

  // Storage for tracking individual digit accuracy and misclassifications
  static int[] class_correct_expect;  // total number of expected correct classifications for each digit (0-9) recorded during last network test
  static int[] class_correct_real;  // number of correct classifications for each digit predicted by network during last network test
  static List<Integer> misclass_idx = new ArrayList<>();  // list of testcase ids (i.e. index in set of all inputs for dataset) that were misclassified during last test (to print misclass images)
  static List<double[]> misclass_activs = new ArrayList<>();  // stores network output corresponding to each misclassified input

  // Create DataSet objects for mnist test data and train data
  static final DataSet mnistTrain = new DataSet("mnist_train.csv", 60000, DataSet.Type.TRAIN);
  static final DataSet mnistTest = new DataSet("mnist_test.csv", 10000, DataSet.Type.TEST);

  // Performs dot product on 2 arrays of equal size and returns resulting scalar
  public static double vDot(double[] a, double[] b){
    assert a.length == b.length : "Incompatible vector sizes for dot product";
    double result = 0.0f;
    for(int i=0; i<a.length; i++) result += a[i]*b[i];
    return result;
  }

  // Performs vector component addition for 2 arrays of equal size and return resulting vector
  public static double[] vAdd(double[] a, double[] b){
    assert a.length == b.length : "Incompatible vector sizes for addition";
    double[] result = new double[a.length];
    for(int i=0; i<a.length; i++) result[i] = a[i]+b[i];
    return result;
  }

  // Sigmoid function used as activation function
  private static double sigma(double z){return 1/(1+(double)Math.exp(-z));}

  // Takes z vector for a given layer and returns an array containing the activations of that layer
  private static double[] getActivations(double[] z){
    double[] result = new double[z.length];
    for(int i=0; i<z.length; i++) result[i] = sigma(z[i]);
    return result;
  }

  // Takes an array of input and returns final layer activations as output
  public static double[] feedForward(double[] input){
    double[][] activations = new double[HIDDEN_LAYERS][HIDDEN_SIZE];  // store activations for all nodes in all hidden layers
    double[] zOut, currWts, aOut;  // store z vals used to compute activations of curr layer, weights to curr layer from prev, activs in curr layer
    double[] output;  // store activations of final layer to return to caller
    double[] zCurr = new double[HIDDEN_SIZE];  // stores z vals for curr layer prior to addition of bias
    
    // Hidden layers
    double[] prev = input;  // previous activations if j=0 are just input
    for(int j=0; j<HIDDEN_LAYERS; j++){  // move through each of the hidden layers...
      for (int k = 0; k < HIDDEN_SIZE; k++) {  // and for each neuron in the curr layer...
        currWts = (j == 0) ? weights_in[k] : weights[j-1][k];  // use w0 if j=0, else use weights[j-1] as the weights to curr layer from previous
        zCurr[k] = vDot(currWts, prev);  // get z vals (w/o bias) from dot product of weights and previous activations
      }
      zOut = vAdd(zCurr, bias[j]); // add biases for curr layer to z vals
      aOut = getActivations(zOut);  // get activations for curr layer
      activations[j] = aOut;  // store activations for curr layer
      prev = aOut;  // activations for curr layer become activations of prev layer
    }

    // Output layer
    zCurr = new double[OUTPUT_SIZE];  // re-init zCurr for number of output nodes
    for(int n=0; n<OUTPUT_SIZE; n++){  // for each node in output layer...
      zCurr[n] = vDot(weights_out[n], prev);  // get z vals (w/o bias) from dot prod of weights for curr node and activs from last hidden layer
    }
    zOut = vAdd(zCurr, bias_out);  // add biases for output layer to get real z vals for output layer
    if(TESTCHECK) System.out.println("\t\t\tz2 = " + Arrays.toString(zOut) + "\n");  // just used to print the final z vals for part1

    output = getActivations(zOut);  // compute/store activations of output layer 

    activs_hid = activations;  // update arrays storing activations of hidden layers
    return output;  // return activations of final output layer to caller
  }

  // Updates counts in digTotals and dig  private void useNetworkInterface(){Correct based on input labels(1 hot vector) and activations
  private static void updateDigitTrack(int testCase, double[] y, double[] aL){
    // Store labeled correct digit as int
    int correct = classifyDigit(y);

    // Store digit associated with highest actication in aL (network's predicted class) as int
    int prediction = classifyDigit(aL); 
    
    // Check if prediction matches correct label
    if(prediction == correct){
      class_correct_real[correct] += 1;  // Update count for correct classifications of the digit if prediction was correct
    }
    // Otherwise add the digit to the current list of misclassifications
    else{
      // Add test case id to list of misclasses
      misclass_idx.add(testCase);
      // store copy of resulting activations in final layer for misclass'd input
      misclass_activs.add(aL.clone());
    }
    // Regardless of prediction, update count for total number of this digit tested
    class_correct_expect[correct] += 1;
  }

  // Perform backwards propagation algorithm on all inputs contained in batch passed as arg
  public static void backPropBatch(double[][] expected_out_batch, double[][] in_batch){
    // init arrays for this batch
    grad_bias_hid = new double[HIDDEN_LAYERS][HIDDEN_SIZE];  // stores bias gradients for each node in each hidden layer
    grad_bias_out = new double[OUTPUT_SIZE];  // stores bias gradients for each node in output layer
    grad_wts_in = new double[HIDDEN_SIZE][INPUT_SIZE];  // stores wt grads for all nodes in hidden <-- input
    grad_wts_out = new double[OUTPUT_SIZE][HIDDEN_SIZE];  // stores wt grads for all nodes in output <-- hidden
    grad_wts_hid = new double[HIDDEN_LAYERS-1][HIDDEN_SIZE][HIDDEN_SIZE];  // stores wt grads for all nodes in all hidden <-- hidden

    // Perform back prop for each set of inputs in the batch
    for(int i=0; i<in_batch.length; i++){
      // store curr set of input
      double[] a0 = in_batch[i];

      // print a0, y, and the training case for debugging
      if(DEBUG){
        System.out.println("\t\tTraining Case " + (i+1) + ":");
        System.out.println("\t\t\ta0 = " + Arrays.toString(a0));
        System.out.println("\t\t\t y = " + Arrays.toString(expected_out_batch[i]) + "\n");
      }

      // store output from feed forward curr set of input
      double[] aL_curr = feedForward(a0);
      // store y for curr set of input
      double[] y_curr = expected_out_batch[i];

      // print activations for debugging
      if(DEBUG){
        for(int j=0; j<HIDDEN_LAYERS; j++){
          System.out.println("\t\t\ta"+(j+1)+" = " + Arrays.toString(activs_hid[j]));
        }
        System.out.println("\t\t\ta" + (HIDDEN_LAYERS+1) + " = " + Arrays.toString(aL_curr));
      }

      // update digit stats for current epoch
      if(DIGIT_TRACK){updateDigitTrack(i, y_curr, aL_curr);}

      // back prop the input
      backPropagate(a0, y_curr, aL_curr);
      
      // print bias and weight gradients during functionality test
      if(TESTCHECK){
        // layer 1
        System.out.println("\n\t\t\tbG1 = " + Arrays.toString(err_hid[0]));
        System.out.println("\t\t\twG1 =\t" + Arrays.toString(wG1[0]));
        for(int j=1; j<3; j++) System.out.println("\t\t\t\t" + Arrays.toString(wG1[j]));
        
        // layer 2
        System.out.println("\n\t\t\tbG2 = " + Arrays.toString(err_out));
        System.out.println("\t\t\twG2 =\t" + Arrays.toString(wG2[0]));
        for(int j=1; j<2; j++) System.out.println("\t\t\t\t" + Arrays.toString(wG2[j]));
        
        System.out.println();
      }
    }

    // Update weights and print them if debugging
    updateWeights(in_batch.length);
    if(DEBUG) System.out.println("\t REVISED WEIGHTS");
    if(DEBUG) printCurrWeights();
  }

  // perform backwards propagation for a single input vector using resulting output activations and label
  private static void backPropagate(double[] input, double[] y_curr, double[] aL_curr){
    // store errors 
    err_hid = new double[HIDDEN_LAYERS][HIDDEN_SIZE];  
    err_out = new double[OUTPUT_SIZE]; 

    // initialize arrays tracking weight gradients if functionality test
    if(TESTCHECK){
      wG1 = new double[3][4];
      wG2 = new double[2][3];
    }
    
    // Output layer
    for(int i=0; i<aL_curr.length; i++){  // for each node in output layer...
      err_out[i] = (aL_curr[i]-y_curr[i])*aL_curr[i]*(1-aL_curr[i]);  // store the activation error of curr node in output layer
      grad_bias_out[i] += err_out[i];  // update sum of bias grad component for curr neuron in output layer
    }
    
    // print cost and error for debugging
    if(DEBUG){
      System.out.println("\n\t\t\tcost = " + cost(aL_curr, y_curr) + "\n");
      System.out.println("\t\t\td" + (HIDDEN_LAYERS+1) + " = " + Arrays.toString(err_out));
    }
    
    // move backwards through each hidden layer, computing/storing error
    for(int l=HIDDEN_LAYERS-1; l>=0; l--){  // for each hidden layer...
      double[] err_prev;  // stores errors from prev (l+1) layer
      double[][] wts_prev;  // stores weights for all nodes in prev layer (l+1) <-- curr layer (l)
      
      // set previous layer according to current layer
      if(l == HIDDEN_LAYERS-1){
        err_prev = err_out;  // if curr layer is final hidden layer: prev layer was output (i.e. error stored in dL)
        wts_prev = weights_out;  // ... weights from curr layer (l) to prev layer (l+1) are in wL
      } else {  // otherwise...
        err_prev = err_hid[l+1];  // prev layer is just l+1 for curr layer l (i.e. error in d[l+1])
        wts_prev = weights[l];  // weights from curr layer l to l+1 are in weights[l]
      }
      
      // compute and store the gradients and error for curr layer
      for(int k=0; k<HIDDEN_SIZE; k++){  // for each of k nodes in curr layer l...
        double weighted_sum_err = 0.0f;  // set current weighted sum of error for curr node k in layer l to l+1
        for(int j=0; j<err_prev.length; j++){  // and for each of j nodes in prev layer l+1...
          if(l == HIDDEN_LAYERS-1){
            grad_wts_out[j][k] += err_prev[j]*activs_hid[l][k];  // update grads for weights from l to l+1 (l+1 = output layer) for curr input
          }
          else{
            grad_wts_hid[l][j][k] += err_prev[j]*activs_hid[l][k];  // update grads for weights from l to l+1 (l+1 = hidden layer) for curr input
          }
          weighted_sum_err += wts_prev[j][k]*err_prev[j];  // update weighted sum of error
        }
        err_hid[l][k] = weighted_sum_err*activs_hid[l][k]*(1-activs_hid[l][k]);  // compute error for node k in current layer l 
        grad_bias_hid[l][k] += err_hid[l][k];  // add computed error for curr input to bias grad
      }
      // Print errors for current layer for debugging
      if(DEBUG) System.out.println("\t\t\td" + (l+1) + " = " + Arrays.toString(err_hid[l]));
    }
    
    // continue moving backwards from 1st hidden layer to input
    double[] err_prev = err_hid[0];  // error for prev layer (l+1) is d[0] when prev layer is 1st hidden layer      
    for(int j=0; j<HIDDEN_SIZE; j++){  // for each node in 1st hidden layer l+1
      for(int k=0; k<INPUT_SIZE; k++){  // and for each node in input layer l
        if(TESTCHECK){
          wG1[j][k] = err_prev[j]*input[k];  // store grad for printing if test network for part1
        }
        grad_wts_in[j][k] += err_prev[j]*input[k];  // update grad for wt to node j in l+1 from node k in l
      }
    }  
  }

  // Computes the cost based on given aL and corresponding label
  private static double cost(double[] out, double[] label){
    double cSum = 0.0f;  // set sum to 0
    for(int i=0; i<out.length; i++){  // for each of the activation vals in out...
      double curr = label[i]-out[i];  // compute diff of network output and label for curr activation
      cSum += curr*curr;  // square the diff and add it to sum
    }
    return 0.5f*cSum;  // cost = 1/2*cSum where cSum = sum of all (label[i]-activation[i])^2 for all i activations/labels 
  }

  // updates the weights in the network according to currently stored gradient vals
  private static void updateWeights(int batchSize){
    double c = ETA/(double)batchSize;  // calculate constant with name I forgot

    // Update weights in w0 using computed constant and weight gradients
    for(int i=0; i<weights_in.length; i++){
      for(int j=0; j<weights_in[i].length; j++){
        weights_in[i][j] -= c*grad_wts_in[i][j];
      }
    }

    // Update the weights in weights for each inner layer
    for(int i=0; i<weights.length; i++){
      for(int j=0; j<weights[i].length; j++){
        for(int k=0; k<weights[i][j].length; k++){
          weights[i][j][k] -= c*grad_wts_hid[i][j][k];
        }
      }
    }

    // Update weights in wL
    for(int i=0; i<weights_out.length; i++){
      for(int j=0; j<weights_out[i].length; j++){
        weights_out[i][j] -= c*grad_wts_out[i][j];
      }
    }

    // Update biases in hidden layers
    for(int i=0; i<bias.length; i++){
      for(int j=0; j<bias[i].length; j++){
        bias[i][j] -= c*grad_bias_hid[i][j];
      }
    }

    // Update biases in output
    for(int i=0; i<bias_out.length; i++){
      bias_out[i] -= c*grad_bias_out[i];
    }
  }

  // Sets all initial weights and biases in network to random val between -1 and 1
  public static void randomizeWeights(){
    Random rand = new Random();

    // Input --> Hidden
    for(int i=0; i<weights_in.length; i++){
      for(int j=0; j<weights_in[i].length; j++){
        weights_in[i][j] = rand.nextDouble()*2-1;
      }
    }

    // Hidden --> Hidden
    for(int i=0; i<weights.length; i++){
      for(int j=0; j<weights[i].length; j++){
        for(int k=0; k<weights[i][j].length; k++){
          weights[i][j][k] = rand.nextDouble()*2-1;
        }
      }
    }

    // Hidden --> Output
    for(int i=0; i<weights_out.length; i++){
      for(int j=0; j<weights_out[i].length; j++){
        weights_out[i][j] = rand.nextDouble()*2-1;
      }
    }

    // Hidden biases
    for(int i=0; i<bias.length; i++){
      for(int j=0; j<bias[i].length; j++){
        bias[i][j] = rand.nextDouble()*2-1;
      }
    }

    // Output biases
    for(int i=0; i<bias_out.length; i++){
      bias_out[i] = rand.nextDouble()*2-1;
    }
  }

  // Fixed test weights for part 1 of assignment
  private static void setTestWeights(){
    double[][] wt0 = {
      {-0.21f, 0.72f, -0.25f, 1.0f},
      {-0.94f, -0.41f, -0.47f, 0.63f},
      {0.15f, 0.55f, -0.49f, -0.75f}
    };

    double[][] b = {{0.1f, -0.36f, -0.31f}};

    double[][] wtL = {
      {0.76f, 0.48f, -0.73f},
      {0.34f, 0.89f, -0.23f}
    };
    double[] bL = {0.16f, -0.46f};

    weights_in = wt0;
    bias = b;
    weights_out = wtL;
    bias_out = bL;
  }

  // Public function called to begin training on a training dataset
  public static void trainSGD(DataSet dataset){
    assert dataset.type == DataSet.Type.TRAIN : "Dataset.Type must be TRAIN";

    if(TESTCHECK){
      testcheckTrain(dataset);  // use hard-coded training algorithm for part 1
    } else {
      int setSize = dataset.getSize();

      Integer[] staticIndxs = new Integer[setSize];  // initialize array to store original order of indices for all sets of input in dataset
      Integer[] indxs = new Integer[setSize];  // init array to store order of indices for the current epoch
      List<Integer> order;  // init list used to randomize order of indices for curr epoch

      for (int i = 0; i < setSize; i++) staticIndxs[i] = i;  // populate array containing indices for all sets of input in dataset
      
      for(int epoch=0; epoch<EPOCHS; epoch++){
        if(DIGIT_TRACK){
          class_correct_expect = new int[10];
          class_correct_real = new int[10];
        }

        order = Arrays.asList(staticIndxs); // create list from array containing original order of indices
        Collections.shuffle(order);  // randomize order of indices contained in list
        order.toArray(indxs);  // store order of input indices to use for curr epoch

        // For all inputs in the training dataset...
        for(int j=0; j<setSize;){  
          // Check that there are enough inputs for the next batch (Don't use if not full batch)
          if((setSize-j) >= BATCH_SIZE){
            double[][] inputBatch = new double[BATCH_SIZE][INPUT_SIZE];  // init array containing the inputs for next batch
            double[][] labelBatch = new double[BATCH_SIZE][OUTPUT_SIZE]; // init array containing the labeled output for next batch
            
            // Select the next BATCH_SIZE number of indices from array of randomized indices and store input/label data at each for next batch
            for(int n=0; n<BATCH_SIZE; n++, j++){
              int idx = indxs[j];  // next random index in array of randomized indices
              inputBatch[n] = dataset.getInputs()[idx];  // store input at next random index
              labelBatch[n] = dataset.getLabels()[idx];  // store labeled output at same next index
            }
            // Perform back propagation using the current batch of inputs and labels
            backPropBatch(labelBatch, inputBatch);
            
          }  
        }
        
        // Print accuracy data per digit and overall accuracy for the current epoch
        if(DIGIT_TRACK){
          System.out.println(String.format("Epoch %d Accuracy:\n", (epoch+1)));
          // Copy curr digit accuracy stats to send to print function
          int[] correctDigs = Arrays.copyOf(class_correct_real, class_correct_real.length);
          int[] totlDigs = Arrays.copyOf(class_correct_expect, class_correct_expect.length);
          printDigitAccuracy(setSize, correctDigs, totlDigs);
        }
      }
    }
  }

  // Print the per digit accuracy stats from the last network test that are stored in the arrays digTotals and digCorrect 
  public static void printDigitAccuracy(int setSize, int[] correctDigs, int[] totalDigs){
    String currStr = "";
    int totlCorrect = 0;  // total number of correct classifications
    double currPercnt;
              
    for(int i=0; i<10; i++){
      // Compute accuracy percentage for current digit
      currPercnt = ((double)correctDigs[i]/totalDigs[i])*100;
      // format/add the current digit and its accuracy to accuracy string
      currStr += String.format("\t%d = %.3f", i, currPercnt);  // add current digit and accuracy percentage
      currStr += "% ";  // add '%' sign
      currStr += String.format("(%d/%d)\n", correctDigs[i], totalDigs[i]);  // add ratio of correct/total for curr digit
      // add number of correct classifications for current digit to total number of correct
      totlCorrect += correctDigs[i];
    }

    // format/add the overall accuracy stats to string
    currPercnt = ((double)totlCorrect/setSize)*100;  // compute overall accuracy
    currStr += String.format("\n\tOverall Accuracy = %.5f", currPercnt);  // add overall percentage
    currStr += "% ";
    currStr += String.format("(%d/%d)\nHorton hears a who!\n", totlCorrect, setSize);  // addratio of correct/total for all inputs
    System.out.println(currStr);  // print the formatted string
  }

  // Feedforward all inputs of a labeled dataset exactly once while recording accuracy stats for each digit and all misclassifications
  private static void getDigitAccuracy(DataSet dataset){
    assert dataset.type == DataSet.Type.TEST || dataset.type == DataSet.Type.TRAIN : "Dataset must be of Type TEST or TRAIN (i.e. labeled) for getDigitAccuracy()";

    double[][] xSet = dataset.getInputs();  // store set of all inputs
    double[][] ySet = dataset.getLabels();  // store set of expected outs
    double[] aL, y;  // init storage for curr output activations and expected output

    // Re-instant the storage arrays for tracking digit accuracy to all 0s
    class_correct_expect = new int[10];
    class_correct_real = new int[10];
    // Clear current list of misclassifications and their corresponding activations
    misclass_idx.clear();
    misclass_activs.clear();
    // For each input in the set of inputs...
    for(int i=0; i<xSet.length; i++){
      aL = feedForward(xSet[i]);  // FF the curr input and store curr output
      y = ySet[i];                // store curr expected output
      updateDigitTrack(i, y, aL);    // update counters according to curr output and expected out
    }
  }

  // Private function called when using the fixed vals for part 1
  private static void testcheckTrain(DataSet dataset){
    double[][] inputBatch = new double[BATCH_SIZE][INPUT_SIZE];
    double[][] labelBatch = new double[BATCH_SIZE][OUTPUT_SIZE];

    for(int i=0; i<EPOCHS; i++){
      // mini-batch 1
      if(DEBUG) System.out.println("\nEpoch " + (i+1) + ":\n\t- Mini-Batch 1:");
      for(int j=0; j<BATCH_SIZE; j++){
        inputBatch[j] = dataset.getInputs()[j];      
        labelBatch[j] = dataset.getLabels()[j];
      }
      backPropBatch(labelBatch, inputBatch);

      // Mini-batch 2
      if(DEBUG) System.out.println("\t- Mini-Batch 2:");
      for(int j=0; j<BATCH_SIZE; j++){
        inputBatch[j] = dataset.getInputs()[j+2];    
        labelBatch[j] = dataset.getLabels()[j+2];
      }
      backPropBatch(labelBatch, inputBatch);
    }
  }

  // Public function called to create the network and initialize all weights, biases, etc.
  public static void createNetwork(){
    if(TESTCHECK) setTestWeights(); 
    else randomizeWeights();
  }

  // Public function which prints the current values for the weights in the network
  public static void printCurrWeights(){
    // Print all weights for input --> hidden (w0)
    System.out.println("\tw1 = \t" + Arrays.toString(weights_in[0])); 
    for(int i=1; i<HIDDEN_SIZE; i++) System.out.println("\t\t" + Arrays.toString(weights_in[i]));

    int l = 0;  // storage for curr layer (index i=l-1)
    for(int i=0; i<HIDDEN_LAYERS; i++){
      l = i+1;  // curr layer is index i+1
      System.out.println("\tb" + l + " = \t" + Arrays.toString(bias[i]));  // print bias in curr layer l
      System.out.println();  // \n

      if(i == HIDDEN_LAYERS-1){  // if curr layer is last hidden layer, weights to l+1 layer is wL
        System.out.println("\tw" + (l+1) + " = \t" + Arrays.toString(weights_out[0]));
        for(int j=1; j<OUTPUT_SIZE; j++){  
          System.out.println("\t\t" + Arrays.toString(weights_out[j]));
        }
      } else {  // otherwise, weights to l+1 layer in weights
        for(int j=0; j<HIDDEN_SIZE; j++){
          System.out.println("\t\t" + Arrays.toString(weights[i][j]));
        }
      }
    }

    ++l;  // move to next layer (update val of l to reflect output layer)
    System.out.println("\tb" + l + " = \t" + Arrays.toString(bias_out) + "\n");  // print biases for output
  }

   /* 
    * WEIGHTS/BIAS STORE ORDER: 
    *  NETWORK_DIMENSIONS...
    *    -->  (INPUT_SIZE,HIDDEN_LAYERS,HIDDEN_SIZE,OUTPUT_SIZE) | 0:3 Columns
    *  WEIGHTS... 
    *    --> w0[0] - ({Input layer > Hidden layer} for neuron n=0) | 0:INPUT_SIZE-1 Columns
    *    --> ... | ...
    *    --> w0[HIDDEN_SIZE-1] - ({Input layer > Hidden layer} for neuron n=HIDDEN_SIZE-1) | 0:INPUT_SIZE-1 Columns
    *    --> weights[0][0] - ({Hidden > Hidden} for hidden layer l=0 and neuron n=0) | 0:HIDDEN_SIZE-1 Columns
    *    --> ... | ...
    *    --> weights[0][HIDDEN_SIZE-1] - ({Hidden > Hidden} for hidden layer l=0 and neuron n=HIDDEN_SIZE-1) | 0:HIDDEN_SIZE-1 Columns
    *    --> weights[1][0] - ({Hidden > Hidden} for hidden layer l=1 and neuron n=0) | 0:HIDDEN_SIZE-1 Columns
    *    --> ... | ...
    *    --> weights[1][HIDDEN_SIZE-1] - ({Hidden > Hidden} for hidden layer l=1 and neuron n=HIDDEN_SIZE-1) | 0:HIDDEN_SIZE-1 Columns
    *    ... ... | ...
    *    * NEW ROW FOR ALL l=HIDDEN_LAYERS-2 LAYERS IN NETWORK *   
    *    ... ... | ...
    *    --> weights[HIDDEN_LAYERS-2][HIDDEN_SIZE-1] - ({Hidden > Hidden} for hidden layer l=HIDDEN_LAYERS-2 and neuron n=HIDDEN_SIZE-1)
    *    --> wL[0] - ({Hidden > Output} for neuron n=0) | 0:HIDDEN_SIZE-1 Columns
    *    --> ... | ...
    *    --> wL[OUTPUT_SIZE-1] - ({Hidden > Output} for neuron n=OUTPUT_SIZE-1) | 0:HIDDEN_SIZE-1 Columns
    *  BIASES
    *    --> bias[0] - (Bias for first hidden layer l=0 for neuron n=0) | 0:HIDDEN_SIZE-1 Columns
    *    --> ... | ...
    *    --> bias[HIDDEN_LAYERS-1] - (Bias for last hidden layer l=HIDDEN_LAYERS-1) | 0:HIDDEN_SIZE-1 Columns
    *    --> bL[0] - (Bias for output layer) | 0:OUTPUT_SIZE-1 Columns
    */
  // Write network config to file as CSV created from filename passed as arg (using write/read order above)
  private static boolean exportWeights(String filename){
    String currDir = String.format("%s/", System.getProperty("user.dir"));  //  get cwd to display to show user location of export dir
    File file = new File(currDir, filename);  // create file obj in currDir with filename arg
    
    assert !file.exists() : String.format("The filename '%s' already exists!", filename);  // tell me if the file already exists in currDir 
    if(!file.exists()){  // if file does not already exist
      try(FileWriter fw = new FileWriter(file)){  // create new filewriter obj for write stream
        // Store network layer dimensions as string of csv and write to file
        String dims =  INPUT_SIZE + "," + HIDDEN_LAYERS + "," + HIDDEN_SIZE + "," + OUTPUT_SIZE + "\n";
        fw.append(dims);  // writes to file

        String delim = ",";  // use ',' as delimeter since csv
        String currRow;  // stores string containing next line of comma seperated vals to write to file

        // Input --> Hidden weights
        for(int i=0; i<weights_in.length; i++){  // for each set of weights in w0
          currRow = "";  // currRow reset to empty string
          for(double val : weights_in[i]){  // for each weight in curr set of weights in w0
            currRow += val + delim;  // add the val of weight to string followed by delimeter
          }
          fw.append(currRow + "\n");  // write currRow (containing each weight in curr set of weights) to file
        }
        
        // Hidden --> Hidden weights
        for(int i=0; i<weights.length; i++){  // for each hidden layer
          for(int j=0; j<weights[i].length; j++){  // for each set of weights in curr layer
            currRow = "";  // reset currRow to empty string
            for(double val : weights[i][j]){  // for each weight in curr set of weights in curr layer...
              currRow += val + delim;  // add weight's val to string followed by delim
            }
            fw.append(currRow + "\n"); // write currRow to file
          }
        }

        // Hidden --> Output weights
        for(int i=0; i<weights_out.length; i++){
          currRow = "";
          for(double val : weights_out[i]){
            currRow += val + delim;
          }
          fw.append(currRow + "\n");
        }
        
        // Hidden biases
        for(int i=0; i<bias.length; i++){ // for each hidden layer's set of bias vals...
          currRow = "";  // reset to empty string
          for(double val : bias[i]){  // for each bias in set of biases for curr layer...
            currRow += val + delim;  // add bias val followed by delim
          }
          fw.append(currRow + "\n");  // write currRow to file
        }
        
        // Output biases
        currRow = "";  // reset to empty string
        for(double val : bias_out){  // for each bias in output layer...
          currRow += val + delim;  // add bias followed by delim to string
        }
        fw.append(currRow + "\n");  // write currRow containing output layer biases to file

        fw.close();  // close write stream
        System.out.println(String.format("Current weights/biases saved to disk as '%s'", filename));  // print success message showing saved filename
        return true;  // return true for successful exort
      }
      catch(IOException e){
        e.printStackTrace();
      }
    }
    return false;  // return false for unsuccessful export
  }
  
  // Restore current network to the config contained in the CSV filename passed as arg (using format above exportWeights())
  private static boolean importWeights(String filename){
    String currDir = String.format("%s/", System.getProperty("user.dir"));  // store cwd to inform user
    File file = new File(currDir, filename);  // create file obj for the file being imported
    assert file.exists() : String.format("'%s' does not contain the specified file '%s'", currDir, filename);  // lemme know if file does not exist in dir
    
    if(file.exists()){  // if the file does exist
      try(BufferedReader br = new BufferedReader(new FileReader(file))){  // create new buffered file reader obj for read stream
        String line;  // stores curr line of file
        String delim = ",";  // delimeter used to separate vals

        // Get network dimensions for network vals contained in file and ensure they match curr network dimensions
        double inSize, hidLayers, hidSize, outSize;  // stores dimensions for the network vals in file
        line = br.readLine();  // read 1st line of file containing network dimensions.
        String[] dims = line.split(delim);  // split the read line into String[] via splitting based on delimeter
        // Parse int val of each numerical string contained in dims array and store as corresponding setting 
        inSize = Integer.parseInt(dims[0]);  // store input size dimension from file
        hidLayers = Integer.parseInt(dims[1]);  // store number of hidden layers from file
        hidSize = Integer.parseInt(dims[2]);  // store number of nodes in each hidden layer from file
        outSize = Integer.parseInt(dims[3]);  // store number of nodes in output layer for file
        // Tell me if any dimension in file does not match the corresponding dimension in current network 
        assert inSize == INPUT_SIZE : "Dimension for 'INPUT_SIZE' being imported does not match current network setting";
        assert hidLayers == HIDDEN_LAYERS : "Dimension for 'HIDDEN_LAYERS' being imported does not match current network setting";
        assert hidSize == HIDDEN_SIZE : "Dimension for 'HIDDEN_SIZE' being imported does not match current network setting";
        assert outSize == OUTPUT_SIZE : "Dimension for 'OUTPUT_SIZE' being imported does not match current network setting";

        // Read through each line in file and store contained vals in appropriate weight/bias arrays

        // Input --> Hidden weights
        weights_in = new double[HIDDEN_SIZE][INPUT_SIZE];  // reset w0 to all 0 weight vals for network dimensions
        for(int i=0; i<HIDDEN_SIZE; i++){  // for each node in hidden layer's set of weights... 
          line = br.readLine();  // read/store next line in file
          String[] tokens = line.split(delim);  // split line into String[] based on delim (each token being a weight val)
          for(int j=0; j<INPUT_SIZE; j++){  // for each node in input layer...
            weights_in[i][j] = Double.parseDouble(tokens[j]);  // parse/store weight to curr node i (in 1st hidden layer) from curr node j (in input layer)
          }
        }

        // Hidden --> Hidden weights
        weights = new double[Math.max(0, HIDDEN_LAYERS-1)][HIDDEN_SIZE][HIDDEN_SIZE];  // reset weights to all 0s for all hidden layers
        for(int l=0; l<HIDDEN_LAYERS-1; l++){  // for each hidden --> hidden layer...
          for(int i=0; i<HIDDEN_SIZE; i++){  // for each node i in layer l+1 
            line = br.readLine();  // read next line in file
            String[] tokens = line.split(delim);  // split
            for(int j=0; j<HIDDEN_SIZE; j++){  // for each node j in layer l
              weights[l][i][j] = Double.parseDouble(tokens[j]);  // parse/store weight to i in l+1 from j in l
            }
          }
        }

        // Hidden --> Output weights
        weights_out = new double[OUTPUT_SIZE][HIDDEN_SIZE];  // reset network weights to 0
        for(int i=0; i<OUTPUT_SIZE; i++){  // for each node i in output..
          line = br.readLine();  // read next line in file
          String[] tokens = line.split(delim);  // split
          for(int j=0; j<HIDDEN_SIZE; j++){  // for each node j in hidden...
            weights_out[i][j] = Double.parseDouble(tokens[j]);  // parse/store weight to i from j
          } 
        }

        // Hidden biases
        bias = new double[HIDDEN_LAYERS][HIDDEN_SIZE];  // reset bias vals for all nodes in hidden layers to 0
        for(int l=0; l<HIDDEN_LAYERS; l++){  // for each hidden layer l
          line = br.readLine();  // read next line in file
          String[] tokens = line.split(delim);  // split
          for(int i=0; i<HIDDEN_SIZE; i++){  // for each node i in curr layer l...
            bias[l][i] = Double.parseDouble(tokens[i]);  // parse/store bias val for node i in layer l
          }
        }

        // Output biases
        bias_out = new double[OUTPUT_SIZE];  // reset bias vals for nodes in output
        line = br.readLine();  // read next line in file (should be final line)
        String[] tokens = line.split(delim);  // split
        for(int i=0; i<OUTPUT_SIZE; i++){  // for each node i in output layer...
          bias_out[i] = Double.parseDouble(tokens[i]);  // parse/store bias val for node i
        }

        br.close();  // close buffered read stream
        System.out.println(String.format("Weights from '%s' imported successfully!", filename));  // print success statement
        return true;  // return true for successful import

      } 
      catch(IOException e){
        e.printStackTrace();
      }
    }
    else{
      System.out.println(String.format("The file '%s' could not be located in the current directory!"));  // print unsuccessful statement
    }
    return false;  // return false for unsuccessful import
  }

  // Main menu interface brought up at program start allowing creation of new network, load prev network, or exit
  public static void startMainInterface(){
    String prompt, currMenu;  // storage for current console prompt to user and string containing the current menu to display
    boolean active = true;  // flag indicating current status of interface (goes false when exiting)
    boolean importSuccess = false;  // flag indicating a network was successfully imported from file (if that selection was made)
    boolean networkReady = false;  // flag indicating that the new network has been trained and is ready for use or imported network is ready for use

   /* 
    *      - MNIST Digit Classifier -
    * Network Selection - Main Menu
    *    [1] : Start New Network to Train
    *    [2] : Load Previously Trained Network
    *
    *    [0] : Exit Interface
    * 
    * Enter Selection: 
    */
    // Store string showing menu options
    String mainMenu = "\t\t - MNIST Digit Classifier -\n\nNetwork Select - Main Menu\n";
    mainMenu += "\t[1] : Start New Network to Train\n";
    mainMenu += "\t[2] : Load Previously Trained Network\n\n";
    mainMenu += "\t[0] : Exit Interface\n\nEnter selection: ";
    try{
      Console console = System.console();  // store reference to console for use

      while(active){  // while the interface is active...
        
        // If a network has already been imported or created/trained show expanded menu to use the current network
        if(networkReady){
          useNetworkInterface();  // call to show expanded menu interface for valid network
        }

        // clear terminal
        clearTerminal();

        // Get user choice as string
        String choice = console.readLine(mainMenu);
        // Parse input as integer
        int val = Integer.parseInt(choice);
        
        // Start new network to train
        if(val == 1){
          clearTerminal();
          randomizeWeights();  // randomize weights/biases
          if(useTrainInterface()){  // the training interface begins training on mnist training set and returns true once training completed, else false
            networkReady = true;  // network has completed training according to number of epochs and thus ready for use
            console.readLine("\nPress ANY Key for Current Network Menu");  // wait for input from user before clearing terminal and moving to next menu
          }
        }
        // Import a network from CSV
        else if(val == 2){
          // Update flag for successful network import to false
          importSuccess = false;  
          // Update string to reflect current menu
          currMenu = "\t\t - MNIST Digit Classifier -\nLoad Pre-Trained Network\n\n";
          // Clear terminal
          clearTerminal();
          // Format string for next user prompt
          prompt = String.format("Enter CSV filename to import from current directory...\n%s/", System.getProperty("user.dir"));
          // Print current menu and prompt user for input
          System.out.println(currMenu);
          String wtsFilename = console.readLine(prompt);  // store user input for filename
          importSuccess = importWeights(wtsFilename);  // update flag with boolean returned from importWeights()

          // update valid network flag if import successful
          if(importSuccess){
            networkReady = true;
          }
          // If import fails, prompt user to re-enter filename or return to main menu
          else{
            // Stay on this import menu until user selects return to main menu or successful network import
            while(!importSuccess){
              clearTerminal();  // clear terminal
              System.out.println(currMenu);  // re-print curr menu title
  
              // Print incorrect CSV filename
              System.out.print(String.format("\n'%s' does not exist in current directory!\n\n", wtsFilename));
  
              // Prompt user to re-enter filename or return to main menu
              choice = console.readLine("Current Menu\n\t[2] : Re-enter CSV Filename\n\t ANY OTHER Returns to Network Selection - Main Menu\n\nEnter Selection: ");
              
              // If user selects to re-enter filename...
              if(Integer.parseInt(choice) == 2){
                clearTerminal();  // clear terminal
                System.out.println(currMenu);  // re-print curr menu
                // Format prompt string asking user for filename
                prompt = String.format("\n\nPlease enter the correct CSV filename:\n%s/", System.getProperty("user.dir"));

                // Prompt user and store CSV filename and attempt import/update to flag
                wtsFilename = console.readLine(prompt);
                importSuccess = importWeights(wtsFilename);

                // If import successful update network ready flag
                if(importSuccess){
                  networkReady = true;
                }
                // If import fails return to top of inner loop conditional on importSuccess flag
              }
              // Otherwise return to top of outer loop conditional on active flag
              else{break;}
            }
          }
        }
        // Exit interface
        else if(val == 0){
          active = false;  // update flag (prob not necessary but just in case)
          System.exit(1);  // exit
        }
        // Otherwise inform user selection invalid
        else{console.printf("User Entered: '%s' - NOT VALID SELECTION!", choice);}
      }
    }
    catch(Exception e){
      e.printStackTrace();
    }
  }

  // Simple interface shown when user selects create new network, allowing user select from starting training, return to network select menu, or exit 
  private static boolean useTrainInterface(){
    String choice;  // stores curr user input resulting from prev prompt

   /* 
    *      - MNIST Digit Classifier -
    * 
    * New Network Menu
    *    [1] : Begin Training on MNIST Training Data
    *    
    *    [9] : Return to Select Network Menu 
    *    [0] : Exit Interface
    * 
    * Enter Selection: 
    */
    // Create string showing menu options
    String netMenu = "\t\t - MNIST Digit Classifier -\n\nNew Network Menu\n";
    netMenu += "\t[1] : Begin Training on MNIST Training Data\n\n";
    netMenu += "\t[9] : Return to Select Network Menu\n";
    netMenu += "\t[0] : Exit Interface\n\n";
    netMenu += "Enter Selection: ";

    try{
      Console console = System.console();  // store console reference for use
      
      while(true){  // infinite loop to remain on this menu until valid selection made
        clearTerminal();  // clear terminal of prev menu
        choice = console.readLine(netMenu);  // print string showing menu as prompt and store user response
        int val = Integer.parseInt(choice);  // parse/store user response as integer (should be int unless invalid selection)

        // start training
        if(val == 1){
          clearTerminal();  // clear prev menu from terminal
          System.out.println("\n\t*** Beginning Training! ***\n");  // print message indicating training has began
          trainSGD(mnistTrain);  // train curr network on mnist training set for specified number of epochs (prints digit/overall accuracy after each epoch)
          System.out.println("\n\t*** Training Complete! ***\n");  // print message indicating number of training epochs were completed
          return true;  // returns true to the call in startMainInterface() for successful training 
        }
        // previous menu
        else if(val == 9){
          return false;  // returns false to the call in startMainInterface() for canceled training
        }
        // exit interface
        else if(val == 0){
          System.exit(1);
        }
      }
    }
    catch(Exception e){
      e.printStackTrace();
    }
    return false;  // all other cases return false
  }

  /* Interface shown once a network is ready for use (trained or imported) allowing... 
   *    - display of network accuracy for both mnist training and testing datasets
   *    - run network on mnist test data showing images/labels for all test inputs
   *    - run network on mnist test data showing images/labels only for misclassified inputs
   *    - save the current configuration of the network
   *    - return to network select menu
   *    - exit
   */ 
  private static void useNetworkInterface(){
    boolean goMainMenu = false;  // flag signaling to return to the network select menu
    boolean exportSuccess = false;  // flag signaling that the current network was successfully written to disk

   /* 
    *      - MNIST Digit Classifier -
    * Current Network Menu
    *    [3] : Display Network Accuracy on MNIST Training Data
    *    [4] : Display Network Accuracy on MNIST Testing Data
    *    [5] : Run Network on MNIST Testing Data (Images/Labels for All Inputs)
    *    [6] : Run Network on MNIST Testing Data (Images/Labels for Misclassified Inputs)
    *    [7] : Save Current Network Configuration
    * 
    *    [9] : Return to Network Select Menu  
    *    [0] : Exit Interface
    * 
    * Enter Selection: 
    */
    // create string containing the menu options to display to user
    String netMenu = "\t\t - MNIST Digit Classifier -\nCurrent Network Menu\n";
    netMenu += "\t[3] : Display Network Accuracy on MNIST Training Data\n";
    netMenu += "\t[4] : Display Network Accuracy on MNIST Testing Data\n";
    netMenu += "\t[5] : Run Network on MNIST Testing Data (Images/Labels for All Inputs)\n";
    netMenu += "\t[6] : Run Network on MNIST Testing Data (Images/Labels for Misclassified Inputs)\n";
    netMenu += "\t[7] : Save Current Network Configuration\n\n";
    netMenu += "\t[9] : Return to Network Select Menu\n";
    netMenu += "\t[0] : Exit Interface\n\n";
    netMenu += "Enter Selection: ";

    String prompt, currMenu, choice;  // storage for prompt to user, string containing curr menu, and user response to prev prompt

    try{
      Console console = System.console();  // store reference to console

      while(!goMainMenu){
        // Clear prev menu
        clearTerminal();

        // Get user selection via prompting string containing menu options to user
        choice = console.readLine(netMenu);
        // Parse/store user choice given as string as an integer 
        int val = Integer.parseInt(choice);

        // Display Accuracy on train data
        if(val == 3){
          // clear prev menu
          clearTerminal();
          // Test entire training dataset
          getDigitAccuracy(mnistTrain);
          // Copy curr digit accuracy stats to send to print function 
          // ** started copying arrays rather than using globals trying to debug divide by zero exception in accuracy stats and never changed back after solving
          int[] correctDigs = class_correct_real.clone();
          int[] totlDigs = class_correct_expect.clone();
          // Print the accuracy result for each digit after testing entire set
          printDigitAccuracy(mnistTrain.getSize(), correctDigs, totlDigs);
          // Wait for input from user before clearing terminal and returning to prev menu
          console.readLine("\nPress ANY Key to Return to Current Network Menu");  
        }
        // Display Accuracy on test data
        else if(val == 4){
          // clear prev menu
          clearTerminal();
          // Test entire testing dataset
          getDigitAccuracy(mnistTest);
          // Copy curr digit accuracy stats to send to print function
          int[] correctDigs = class_correct_real.clone();
          int[] totlDigs = class_correct_expect.clone();
          // Print the accuracy results for each digit after testing entire set
          printDigitAccuracy(mnistTest.getSize(), correctDigs, totlDigs);
          console.readLine("\nPress ANY Key to Return to Current Network Menu");  
        }
        // Run network on test data while showing all images/labels
        else if(val == 5){
          // clear prev menu
          clearTerminal();
          // test mnist test dataset showing labels and images for each input
          runDigitTest(mnistTest);
          // wait for input from user before returning to prev menu
          console.readLine("\nPress ANY Key to Return to Current Network Menu");  
        }
        // Run network on test data and show only misclassified digits
        else if(val == 6){
          // clear prev menu
          clearTerminal();
          // run network on mnist test data showing misclass'd (prompting user to continue or return to menu after each)
          getMisclasses();
        }
        // Export network configs
        else if(val == 7){
          exportSuccess = false;  // reset flag to false
          
          // store strings for curr sub-menu title and prompt
          currMenu = "\t\t - MNIST Digit Classifier -\nSave Current Network\n\n\n\n";
          prompt = "\nEnter filename to save the current network as (e.g. 'myNetwork.csv')...\nSave as: ";
          
          // clear prev menu
          clearTerminal();
          System.out.print(currMenu);  // display sub-menu title
          String wtsFilename = console.readLine(prompt);  // prompt user for filename to save file as and store response
          exportSuccess = exportWeights(wtsFilename);  // attempt to write network config to disk (true: successful export, false: unsuccessful export)
          if(exportSuccess){
            clearTerminal();  // clear prev menu
            System.out.println(currMenu);  // re-print curr menu title
            System.out.println(String.format("Successfully Saved Current Network Configuration as '%s'!", wtsFilename));  // print success
            console.readLine("\nPress ANY Key to Return to Current Network Menu");  // wait for user input to return to prev menu
          }

          while(!exportSuccess){  // while the network has not been saved to disk...
            clearTerminal();  // clear prev menu
            System.out.print(currMenu);  // re-print sub-menu title
            // inform user that network was not saved
            System.out.println(String.format("COULD NOT SAVE NETWORK AS '%s'!\nEnsure file does not already exist in directory.\n", wtsFilename));
            // prompt user to either input 7 to re-enter filename and try again, all other input will return to menu for current network
            choice = console.readLine("Current Network Menu\n\t[7] : Re-enter Filename to Save Network As\n\tANY Other Returns to Current Network Menu\n\nEnter Selection: ");
            // if user selects inputs 7 to re-enter the filename...
            if(Integer.parseInt(choice) == 7){
              clearTerminal();  // clear prev menu
              System.out.println(currMenu);  // re-print sub-menu title
              // create string asking user to re-enter a filename
              prompt = "\nRe-enter filename to save current network as (e.g. 'myNetwork.csv')...\nSave as: ";

              // Prompt user and store filename then use to attempt export again
              wtsFilename = console.readLine(prompt);
              exportSuccess = exportWeights(wtsFilename);  // flag updates to true if export successful, else return to top of inner while loop
              if(exportSuccess){
                clearTerminal();  // clear prev menu
                System.out.println(currMenu);  // re-print curr menu title
                System.out.println(String.format("Successfully Saved Current Network Configuration as '%s'!", wtsFilename));  // print success
                console.readLine("\nPress ANY Key to Return to Current Network Menu");  // wait for user input to return to prev menu
              }
            }
            // all other user input returns to prev menu, so break out of inner while loop (returning to top of outer while loop, i.e. prev menu)
            else{break;}
          }          
        }
        // Return to main menu
        else if(val == 9){
          // update flag to true which ends outer while loop and returns to caller startMainInterface() (i.e. returns to displaying Network Selection Menu)
          goMainMenu = true;
        }
        // Exit
        else if(val == 0){
          goMainMenu = true;  // not really needed but just to cover all bases
          System.exit(1);
        }
        // Otherwise inform user selection was invalid
        else{console.printf("User Entered: '%s' - NOT VALID SELECTION!", choice);}
      }
    }
    catch(Exception e){
      e.printStackTrace();
    }

    return;  // return to caller in startMainInterface()
  }

  // Clears current terminal window when called
  private static void clearTerminal(){
    System.out.print("\033[H");  // escape seq to move cursor to top left of terminal window 
    System.out.print("\033[2J");  // ecape seq to clear text after cursor
    System.out.flush();  // flush stream to send to stdout
  }

  // Runs through all inputs in the given labeled dataset printing: test case, image of input, network class, label, accuracy for each input
  private static void runDigitTest(DataSet dataset){
    // If dataset does not have type TEST or TRAIN, its not labeled and can not be used, so need to know if this occurs somehow 
    assert dataset.type == DataSet.Type.TEST || dataset.type == DataSet.Type.TRAIN : "Dataset must be of Type TEST or TRAIN (i.e. labeled) for runDigitTest()";

    Console console = System.console();  // store ref to console for use

    double[][] xSet = dataset.getInputs();  // store set of all inputs
    double[][] ySet = dataset.getLabels();  // store set of expected outs
    double[] aL, x, y, currDist;  // init storage for curr output activations and expected output
    int currPick, currLabel;  // storage for class predicted by network and labeled class

    for(int i=0; i<xSet.length; i++){  // for each input in set of all inputs
      x = xSet[i];  // store curr input vector
      y = ySet[i];  // store curr label vector
      aL = feedForward(x);  // FF curr input and store output activations

      updateDigitTrack(i, y, aL);  // update count for curr digit

      String currArt = getDigitArt(x);  // pass curr input to get string containing art representation of digit used as curr input
      String out = "\n";  // create string to contain results of testing curr input
      currPick = classifyDigit(aL);  // classify the output activations vector to get the integer val of digit predicted by network
      currLabel = classifyDigit(y);  // classify label vector to get int val the current digit was labeled as
      
      out += String.format("Test Case %d ", i);  // add formatted string containing test case to the output string containing details for curr input
      out += (currPick == currLabel) ? "CORRECT:\n" : "INCORRECT:\n";  // add accuracy of prediction to output (correct or incorrect)
      out += String.format("\tNetwork Class = %d | Label Class = %d\n", currPick, currLabel);  // add the network's classification and label's classification
      
      System.out.println(out);  // print the test result details for curr digit
      System.out.println(currArt);  // print digit image for curr digit
      System.out.println("==================================================");  // print line to show split between inputs 

      // Get/print softmax distribution of aL for curr digit if SHOW_DIST is true
      if(SHOW_DIST){
        currDist = getSoftmaxDist(aL);
        out = "\n\tOutput Distribution:\n";
        for(int j=0; j<currDist.length; j++){
          out += String.format("\t\t%d - %.3f\n", j, (currDist[j]*100));
        }
        System.out.println(out);
      }

      //TODO: I THINK SUPPOSED TO WAIT FOR USER INPUT BEFORE MOVING TO NEXT INPUT, BUT LEAVING THIS HERE SO I CAN FIND TO COMMENT OUT FOR TESTING
      // prompt user for input to continue or return to prev menu
      String choice = console.readLine("[1] : Continue\t\tANY OTHER : Previous Menu\nEnter Selection: ");
      // return to prev menu if input not 1
      if(Integer.parseInt(choice) != 1){
        return;
      }  
      
    }
  }

  // Returns int val for digit associated with the highest activation value in array of activations given as input 
  private static int classifyDigit(double[] aL){
    int pick = 0;  // assume that 0 has highest activation val to start
    for(int i=1; i<aL.length; i++){  // for each of the remaining digits (1-9)...
      pick = (aL[i] > aL[pick]) ? i : pick;  // For curr digit (i), if aL[i] > aL[pick] then i is new pick and aL[i] is now greatest activation val
    }
    // return the digit associated with the highest activation val after checking all activation vals in aL 
    return pick;
  }

  // Returns array containing softmax distribution for the array of activations given as input
  private static double[] getSoftmaxDist(double[] aL){
    double[] dist = new double[aL.length];  // create array to store softmax distribution
    double eSum = 0;  // use to compute sum of e^(x_i) for all x_i in aL

    // compute denominator sum used to compute each component of the distribution
    for(int i=0; i<aL.length; i++){eSum += Math.exp(aL[i]);}

    // for each of i components in aL: the i'th component in distribution has value of e^(aL[i])/eSum so compute/store each component
    for(int i=0; i<aL.length; i++){
      dist[i] = Math.exp(aL[i])/eSum;
    }

    return dist;  // return the resulting softmax distribution to caller
  }

  // Takes array of grayscale pixel vals normalized b/t 0-1 for 28x28 pixel image and returns string containing an ascii representation of the image
  private static String getDigitArt(double[] pixelVals){
    assert pixelVals.length == 784 : "Array containing normalized pixel values is not 28x28!";  // check that array has exactly 784 vals (28x28)

    /*
     * pixel vals b/t 0-1... 
     *    => where on shades of darkness scale curr pixel is (0=white, 1=black)
     *    => if chars arranged to represent shades of darkness scale, location of chars within that scale correspond to shades of pixel vals b/t 0-1
     * index of ascii char corresponding to shade of curr pixel val computed via multiplying pixel val by length of char darkness scale
     *    => when pixel val is 1, index will be last char in char darkness scale (i.e. length-1)
     *    => when pixel val is 0, index will be first char in char darkness scale (i.e. 0)
     * 
     */
    
    String art = "";  // create string for ascii art
    char[] charScale = {' ', '.', '*', ':', 'o', '&', '8', '#', '@' };  // chars representing shades of grayscale for 'darkness scale' (light --> dark) made from ascii chars 
    
    // height and weight of image
    int height = 28;
    int width = 28;
    
    int idx;  // stores index of appropriate char in char darkness scale corresponding to the current pixels grayscale val

    int k=0;  // tracks curr pixel who's val is being used to find its corresponding char  
    for(int i=0; i<height; i++){  // for each line of pixels in height of image...
      for(int j=0; j<width; j++){  // for each pixel moving from left to right in curr line
        idx = (int)(pixelVals[k++]*(charScale.length-1));  // compute where along the character darkness scale the appropriate ascii char is for curr pixel k
        art += charScale[idx];  // add char at same relative point along char darkness scale as the curr pixel is along the white to black scale
      }
      art += "\n";  // move to next line in image
    }
    return art;  // return string containing image to caller
  }

  // Run mnist testing set through current network and show images/labels for all digits misclassified by the network
  private static void getMisclasses(){
    double[][] xSet = mnistTest.getInputs();  // store set of all inputs
    double[][] ySet = mnistTest.getLabels();  // store set of expected outs
    double[] x, curr_aL;  // init storage for curr input and curr final activations
    int currPick, currLabel, idx;  // storage for network prediction, label, id (i.e. index for input in set of inputs)
    String currArt, out, choice;  // storage for strings containing: rep of curr digit (art), digit descriptions, user input 

    // Run through testing set, recording misclassifications and stats
    getDigitAccuracy(mnistTest);
    
    try{
      Console console =  System.console();
      
      // For each misclassified digit, print: id, image, predict, label
      for(int i=0; i<misclass_idx.size(); i++){
        clearTerminal();  // clear terminal
        curr_aL = misclass_activs.get(i);  // get network output for current misclass'd digit
        idx = misclass_idx.get(i);  // get index for misclass'd input in set of all inputs
        
        x = xSet[idx];  // get input for current misclass'd digit 
        currArt = getDigitArt(x);  // get string rep for misclass'd digit
        
        currPick = classifyDigit(curr_aL);  // digit predicted by network
        currLabel = classifyDigit(ySet[idx]);  // correct label of digit
        
        // Format string to print to terminal with test case id, network's class, and correct class
        out = "\t\t - MNIST Digit Classifier -\nMisclassifications\n\n";
        out += String.format("Test Case %d INCORRECT:\n", idx);
        out += String.format("\tNetwork Class = %d | Label Class = %d\n", currPick, currLabel);

        System.out.println(out);  // print formatted string
        System.out.println(currArt);  // print representation of digit
        choice = console.readLine("[1] : Continue\t\tANY OTHER : Previous Menu\nEnter Selection: ");  // prompt user for input to continue or return to prev menu

        // Break loop to return to calling function (previous menu) if user input is not 1
        if(Integer.parseInt(choice) != 1){break;}
      }
    }
    catch(Exception e){
      e.printStackTrace();
    }
    return;  // return to calling function
  }

  // Class created to store data to be given to network
  public static class DataSet{
    public enum Type { TEST, TRAIN, PROD }  // enum for types of datasets for testing, training, and production  

    String filename;  // filename for the csv containing data which makes up the data set
    Type type;  // data set type enum
    int size;  // number of inputs/data points contained in data set 
    double[][] labels;  // stores the labels associated with inputs for labeled data sets
    double[][] inputs;  // stores all the inputs in data set

    // Constructor
    DataSet(String filename, int numSamples, Type type){
      this.filename = filename;
      this.size = numSamples;
      this.type = type;

      // if labeled data set, create array to store labels based on size of data set and output size, then read csv in filename as labeled data
      if(this.type == Type.TEST || this.type == Type.TRAIN){
        this.labels = new double[this.size][OUTPUT_SIZE];
        readLabeledData();
      } 
      // otherwise, set label array to null and read csv in filename as unlabeled data
      else{
        this.labels = null;
        readData();
      }
    }

    // Constructor with no args (for the initial functionality test check in part 1) using hard-coded vals
    DataSet(){
      this.filename = "NETWORK TEST";
      this.size = 4;
      this.type = Type.TRAIN;

      double[][] inputs = {
        {0, 1, 0, 1},
        {1, 0, 1, 0},
        {0, 0, 1, 1},
        {1, 1, 0, 0}
      };

      double[][] labels = {
        {0, 1},
        {1, 0},
        {0, 1},
        {1, 0}
      };

      defineInputs(inputs);
      defineLabels(labels);
    }

    // setters
    public void defineInputs(double[][] inputs){this.inputs = inputs;}
    public void defineLabels(double[][] labels){this.labels = labels;}
    // getters
    public int getSize(){return this.size;}
    public double[][] getInputs(){return this.inputs;}
    public double[][] getLabels(){return this.labels;}

    // reads unlabeled data
    private void readData(){
      try(BufferedReader br = new BufferedReader(new FileReader(this.filename))){
        String line;  // stores string containing curr line read from file
        String delim = ",";  // delimeter used to separate vals
        List<double[]> inputs = new ArrayList<>();  // stores double[] containing vals making up each input in dataset while reading from file

        while((line = br.readLine()) != null){  // read the current line and store it, if the line is not null...
          String[] tokens = line.split(delim);  // split the line based on delim
          double[] vals = new double[INPUT_SIZE];  // create array to store vals making up input contained on curr line
          for(int i=0; i<INPUT_SIZE; i++){  // for each component of the input, store the parsed int val after normalizing
            vals[i] = Integer.parseInt(tokens[i])/255.0f;  // normalize pixel greyscale val and store as double by dividing val by max (255) as double/float
          }
          inputs.add(vals);  // store normalized input from curr line 
        }
        this.defineInputs(inputs.toArray(new double[inputs.size()][INPUT_SIZE]));  // create array from list of inputs and store as instanced variable of dataset
        this.size = inputs.size();  // set the dataset size
      } catch(IOException e){
        e.printStackTrace();
      }
    }

    // reads labeled data same as unlabeled, except that label corresponding to each input is stored in separate list and then stored for the dataset
    private void readLabeledData(){
      try(BufferedReader br = new BufferedReader(new FileReader(this.filename))){
        String line;
        String delim = ",";
        List<double[]> inputs = new ArrayList<>();
        List<double[]> labels = new ArrayList<>();

        while((line = br.readLine()) != null){
          String[] tokens = line.split(delim);
          double[] vals = new double[INPUT_SIZE];

          for(int i=0; i<(tokens.length-1); i++){
            vals[i] = Integer.parseInt(tokens[i+1])/255.0f;
          }
          labels.add(oneHotVector(Integer.parseInt(tokens[0]), OUTPUT_SIZE));
          inputs.add(vals);
        }
        this.defineInputs(inputs.toArray(new double[inputs.size()][INPUT_SIZE]));
        this.defineLabels(labels.toArray(new double[labels.size()][OUTPUT_SIZE]));
        this.size = inputs.size();
      } catch(IOException e){
        e.printStackTrace();
      }
    }

    // creates 1 hot vector for the digit label given as an integer
    private static double[] oneHotVector(int label, int outputSize){
      double[] vector = new double[outputSize];
      // bounds check (avoids AIOOB except if label unexpected val)
      if (label >= 0 && label < outputSize) vector[label] = 1.0f;
      return vector;
    }
  }
}

