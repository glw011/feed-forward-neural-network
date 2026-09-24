package source.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.Optional;

public class DataSet {
      

    String name;
    String filepath;  // filename for the csv containing data which makes up the data set
    DataSetType type;  // data set type enum
    int size;  // number of inputs/data points contained in data set 
    double[][] inputs;  // stores all the inputs in data set
    double[][] labels;  // stores the labels associated with inputs for labeled data sets

    Integer[] indices;

    
    DataSet(String name, DataSetType type, int outputSize, String filepath) throws Exception {
        this.name = name;
        this.filepath = filepath;
        this.type = type;

        DataInputs data = parseFileData(filepath, type, outputSize);

        if(data.labels() == null && this.type != DataSetType.PROD)
                throw new Exception(String.format("ERROR - failed to parse '%s' as labeled data; verify data integrity", filepath));

        this.inputs = data.inputs();
        this.labels = data.labels();
        this.size = this.inputs.length;

        this.indices = new Integer[this.size];
        for(int i=0; i<this.indices.length; i++) this.indices[i] = i;
    }

    DataSet(String name, DataSetType type, double[][] inputs, double[][] labels) {
        this.name = name;
        this.filepath = null;
        this.type = type;
        this.inputs = inputs;
        this.labels = labels;
        this.size = inputs.length;

        this.indices = new Integer[this.size];
        for(int i=0; i<this.indices.length; i++) this.indices[i] = i;
    }

    /* TODO: Temp no-arg constructor DELETE ONCE DataSet fully implemented */
    DataSet() {}

    public DataSet newDataSet(String name, DataSetType type, double[][] inputs, double[][] labels) throws Exception {
        return new DataSet(name, type, inputs, labels);
    }

    public DataSet newDataSetFromFile(String name, DataSetType type, int outputSize, String filepath) throws Exception {
        return new DataSet(name, type, outputSize, filepath);
    }

    /* TODO: needs to create & return TRAIN type DataSet created from default files in `/resources/mnist/` once they are there */
    public static DataSet defaultMnistTrainingSet() {return new DataSet();}
    
    /* TODO: needs to create & return TEST type DataSet created from default files in `/resources/mnist/` once they are there */
    public static DataSet defaultMnistTestingSet() {return new DataSet();}

    public void defineInputs(double[][] inputs) {this.inputs = inputs;}
    public void defineLabels(double[][] labels) {this.labels = labels;}
    public int size() {return this.size;}
    public double[][] inputs() {return this.inputs;}
    public double[][] labels() {return this.labels;}
    public Integer[] indices() {return this.indices;}

    // reads unlabeled data as Doubles, applies the passed normalize function to each parsed val, and returns double[][] containing results 
    private static double[][] readUnlabeledDataFromCsv(String filename, Function<Double, Double> normalize) throws Exception {
        try(BufferedReader br = new BufferedReader(new FileReader(filename))){
            String line;                                                            // stores curr line read from file
            String delim = ",";                                                     // delimeter used to separate vals
            List<double[]> inputs = new ArrayList<>();

            double[] vals;
            while((line = br.readLine()) != null){                                  // read curr line and store if not null...
                String[] tokens = line.split(delim);                                // split line based on delim
                vals = new double[tokens.length];                                   // array storing input vals from curr line

                for(int i=0; i<vals.length; i++){
                    vals[i] = normalize.apply((Double.parseDouble(tokens[i])));     // store ea parsed val after normalizing
                }

                inputs.add(vals);                                                   // store array of normalized vals from curr line
            }
            return inputs.toArray(new double[inputs.size()][inputs.get(0).length]);
        } 
    }

    // reads labeled data same as unlabeled, except that label corresponding to each input is stored in separate list and then stored for the dataset
    private static DataInputs readLabeledDataFromCsv(String filename, int outputSize, Function<Double, Double> normalize) throws Exception {
        try(BufferedReader br = new BufferedReader(new FileReader(filename))){
            String line;
            String delim = ",";
            List<double[]> inputs = new ArrayList<>();
            List<double[]> labels = new ArrayList<>();

            double[] vals;
            while((line = br.readLine()) != null){
                String[] tokens = line.split(delim);
                vals = new double[tokens.length-1];

                for(int i=0; i<(tokens.length-1); i++){
                    vals[i] = normalize.apply((Double.parseDouble(tokens[i+1])));
                }

                labels.add(encodeOneHotVector(Integer.parseInt(tokens[0]), outputSize));
                inputs.add(vals);
            }

            return new DataInputs(
                    inputs.toArray(new double[inputs.size()][inputs.get(0).length]), 
                    labels.toArray(new double[labels.size()][outputSize]));
        }
    }

    private record DataInputs(double[][] inputs, double[][] labels) {}

    // encodes 1 hot vector from integer label and returns resulting vector as double[]
    private static double[] encodeOneHotVector(int label, int outputSize) throws Exception{
        double[] vector = new double[outputSize];
        if (label >= 0 && label < outputSize) vector[label] = 1.0f;
        else throw new Exception(String.format("Parse failure; unexpected label value: %d", label));
        return vector;
    }

    private static Optional<String> getExtension(String filename) {
        if(filename.isBlank() || filename == null) return Optional.empty();

        int dotIdx = filename.lastIndexOf((int)'.');
        if(dotIdx > 0 && dotIdx < filename.length()-1) {
            return Optional.of(filename.substring(dotIdx, filename.length()));
        }

        return Optional.empty();
    }

    private static DataInputs parseFileData(String filepath, DataSetType type, int outputSize) throws Exception {
        String ext = getExtension(filepath)
                        .orElseThrow( () -> new Exception(
                                    String.format("ERROR - failed to parse filepath: '%s'", filepath)));

        if(ext.equals(FileType.CSV.ext())){
            return getCsvFileData(filepath, type, outputSize);
        } else if(ext.equals(FileType.JSON.ext())) {
            return getJsonFileData(filepath, type, outputSize);
        }
        
        throw new Exception(
                String.format("ERROR - failed to parse data in '%s'; *%s is not supported", 
                                filepath, ext));       
    }
        
    private static DataInputs getCsvFileData(String filepath, DataSetType type, int outputSize) throws Exception {
        if(type == DataSetType.TEST || type == DataSetType.TRAIN) {
            return readLabeledDataFromCsv(filepath, outputSize, (n) -> {return n/255.0f;});
        } else {
            return new DataInputs(
                        readUnlabeledDataFromCsv(filepath, (n) -> {return n/255.0f;}), null);
        }
    }
        
    private static DataInputs getJsonFileData(String filepath, DataSetType type, int outputSize) throws Exception {
        throw new Exception(
                String.format("ERROR - failed to parse data in '%s'; *.json format currently unsupported", 
                                    filepath));
    }
        
}
