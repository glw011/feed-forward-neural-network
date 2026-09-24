package source.network;

import source.data.DataSet;

public class NetworkContainer {
    NeuralNetwork network;
    DataSet trainingData;
    DataSet testingData;

    public NetworkContainer() {
        this.network = null;
        this.trainingData = null;
        this.testingData = null;
    }

    public NetworkContainer(NeuralNetwork network, DataSet trainingData, DataSet testingData) {
        this.network = network;
        this.trainingData = trainingData;
        this.testingData = testingData;
    }

    public NeuralNetwork network() {return this.network;}
    public DataSet trainingData() {return this.trainingData;}
    public DataSet testingData() {return this.testingData;}

    public void setNetwork(NeuralNetwork network) {this.network = network;}
    public void setTrainingData(DataSet trainingData) {this.trainingData = trainingData;}
    public void setTestingData(DataSet testingData) {this.testingData = testingData;}
}
