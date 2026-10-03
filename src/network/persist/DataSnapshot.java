package src.network.persist;

import src.network.data.DataSet;

public record DataSnapshot(DataSet trainingData, DataSet testingData) {}
