package src.network.components;

import src.network.data.DataSet;

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
    
    public boolean loadNetwork(String networkFilename) {
        /* TODO: Not implemented 
                method is passed a valid "mynet.nn" filename 
                needs to: 
                    import/create NeuralNetwork importedNetwork from that file 
                    this.setNetwork(importedNetwork)
                    return true
                otherwise
                    return false 
        */
       return false;
    }
}
