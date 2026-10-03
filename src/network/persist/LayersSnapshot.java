package src.network.persist;

import src.network.components.Layer;

public record LayersSnapshot(Layer[] hiddenLayers, Layer outputLayer) {}
