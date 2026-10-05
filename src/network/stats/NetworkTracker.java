package src.network.stats;

import java.util.ArrayList;
import java.util.List;


public class NetworkTracker {
    int[] totalActualCorrect;
    int[] totalExpectedCorrect;

    int[][] epochsActualCorrect;
    int[][] epochsExpectedCorrect;

    boolean isTrackingEpochAccuracy;

    List<Misclassification> misclassCases;

    NetworkTracker(int outputSize, int epochs) {
        this.totalActualCorrect = new int[outputSize];
        this.totalExpectedCorrect = new int[outputSize];
        
        this.epochsActualCorrect = new int[epochs][outputSize];
        this.epochsExpectedCorrect = new int[epochs][outputSize];

        this.isTrackingEpochAccuracy = true;

        this.misclassCases = new ArrayList<>();
    }

    NetworkTracker(int outputSize) {
        this.totalActualCorrect = new int[outputSize];
        this.totalExpectedCorrect = new int[outputSize];
        
        this.epochsActualCorrect = null;
        this.epochsExpectedCorrect = null;

        this.isTrackingEpochAccuracy = false;

        this.misclassCases = new ArrayList<>();
    }

    public void countClassifyMiss(int expectedIdx, int actualIdx, double[] activations, int caseIdx) {
        this.totalExpectedCorrect[expectedIdx] += 1;
        this.misclassCases.add(new Misclassification(caseIdx, activations.clone()));
    }

    public void countTrainClassifyMiss(int expectedIdx, int actualIdx, double[] activations, int caseIdx, int epoch) {
        this.countClassifyMiss(expectedIdx, actualIdx, activations, caseIdx);

        if(isTrackingEpochAccuracy 
                    && this.epochsExpectedCorrect != null 
                    && this.epochsActualCorrect != null) {
            this.epochsExpectedCorrect[epoch][expectedIdx] += 1;
        }
    }

    public void countClassifyCorrect(int correctIdx) {
        this.totalExpectedCorrect[correctIdx] += 1;
        this.totalActualCorrect[correctIdx] += 1;
    }

    public void countTrainClassifyCorrect(int correctIdx, int epoch) {
        this.countClassifyCorrect(correctIdx);

        if(isTrackingEpochAccuracy 
                    && this.epochsExpectedCorrect != null 
                    && this.epochsActualCorrect != null) {
            this.epochsExpectedCorrect[epoch][correctIdx] += 1;
            this.epochsActualCorrect[epoch][correctIdx] += 1;
        }
    }

    public void clear() {
        clearCountsArray(this.totalActualCorrect);
        clearCountsArray(this.totalExpectedCorrect);
        this.misclassCases.clear();

        if(this.epochsActualCorrect != null)
            for(int i=0; i < epochsActualCorrect.length; i++)
                clearCountsArray(epochsActualCorrect[i]);

        if(this.epochsExpectedCorrect != null)
            for(int i=0; i < epochsExpectedCorrect.length; i++)
                clearCountsArray(epochsExpectedCorrect[i]);
    }

    private static void clearCountsArray(int[] counts) {
        for(int i=0; i<counts.length; i++) counts[i] = 0;
    }

    public int[] totalActualCorrect() {return this.totalActualCorrect;}

    public int[] totalExpectedCorrect() {return this.totalExpectedCorrect;}

    public int[][] epochsActualCorrect() {return this.epochsActualCorrect;}

    public int[][] epochsExpectedCorrect() {return this.epochsExpectedCorrect;}

    public static NetworkTracker newTracker(int outputSize) {
        return new NetworkTracker(outputSize);
    }

    public static NetworkTracker newTrainingTracker(int outputSize, int totalEpochs) {
        return new NetworkTracker(outputSize, totalEpochs);
    }
}
