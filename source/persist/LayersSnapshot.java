package source.persist;

import source.network.Layer;

public record LayersSnapshot(Layer[] hiddenLayers, Layer outputLayer) {}
