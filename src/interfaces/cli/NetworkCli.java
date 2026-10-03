package src.interfaces.cli;

import java.io.Console;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import src.network.config.NetworkConfig;
import src.network.data.DataSet;
import src.network.data.DataSetType;
import src.network.components.NeuralNetwork;
import src.network.components.NetworkContainer;

 /*
    USER STORIES
    - As a user I'd like to recieve a prompt notifying me I will lose the current network configuration after selecting return to the network select menu 
      or exiting the interface if a custom network has been created but has not yet been saved.
      * implement `customNetworkIsUnsaved` flag  and relevant checks when `status = 9` is returned in `neuralNetworkCli()`
 */

public class NetworkCli {

    static List<Path> savedNetworks = null;

    private enum MenuState {
        NET_SELECT,
        NEW_NET,
        NETWORK,
        LOAD_NET,
        CUSTOM_NET,
        SET_DATA,
        TRAIN,
    }

    public static void startNeuralNetCli() {
        int status = -1;

        Console console = null;
        try {
            console = System.console();
            if(console == null) throw new IOException("The JVM console was returned as null!");
        } catch(IOException e) {
            System.out.println("ERROR: " + e.getMessage());
            System.out.println("\nThis can often be caused by running this application within certain IDE's built-in terminal.");
            System.out.println("\nPlease run the application from a native terminal");
            System.exit(0);
        }

        MenuState menu = MenuState.NET_SELECT;

        NetworkContainer neuralNet = new NetworkContainer();

        while(true) {
            if(menu == MenuState.NET_SELECT) {
                status = netSelectMenu(console);
                if(status == 1) {
                    //neuralNet = initDefaultNetwork();
                    menu = MenuState.NEW_NET;
                } else if(status == 2) {
                    menu = MenuState.LOAD_NET;
                }
            } else if(menu == MenuState.NEW_NET) {
                status = newNetworkMenu(console);
                if(status == 1) {
                    neuralNet = initDefaultNetwork();
                    menu = MenuState.NETWORK;
                } else if(status == 2) {
                    menu = MenuState.CUSTOM_NET;
                }
            } else if(menu == MenuState.CUSTOM_NET) {
                status = newCustomNetMenu(console, neuralNet);
                if(status == 1) {
                    if(neuralNet.network() == null) 
                        throw new UnknownError("custom network was successfully configured but is null");
                    
                    // 1 is returned && network is not null  -->  network configured successfully and ready to set datasets
                    menu = MenuState.SET_DATA;
                }
            } else if(menu == MenuState.LOAD_NET) {
                status = loadNetworkMenu(console, neuralNet);
                if(status == 1) {
                    if(neuralNet.network() == null) 
                        throw new UnknownError("custom network was successfully configured but is null");
                    
                    // 1 is returned && network is not null  -->  network configured successfully and ready to set datasets
                    menu = MenuState.SET_DATA;
                }
            } else if(menu == MenuState.SET_DATA) {
                status = selectDatasetsMenu(console, neuralNet);
                
                if(status == 1) {
                    if(neuralNet.trainingData() == null || neuralNet.testingData() == null)
                        throw new UnknownError("DataSets were successfully selected but 1 or more is null");

                    menu = MenuState.NETWORK;
                } else if(status == 2) {
                    menu = MenuState.SET_DATA;
                }
            } else if(menu == MenuState.NETWORK) {
                status = networkMenu(console);
                /* TODO: not finished */
            }

            if(status == 9) menu = MenuState.NET_SELECT;
            else if(status == 0) System.exit(status);
            else if(status < 0) throw new UnknownError();
        }
    }

    /*
      Network menu
      [1] Network Testing submenu
            * Test Network (Using Current Testing Config)               **REQUIRES TRAINING STATUS: TRAINED**
            * Previous Test's Overall Accuracy Stats
            * Current Testing Configuration
                + Testing Dataset
                + Image/Label Setting
            * Adjust Testing Configuration
                + Change Image/Label Setting
                    - Images/Labels for all inputs
                    - Images/Labels only for misclassified inputs
                    - No Images/Labels  
                + Change Testing Dataset
                    - Choose Previously Saved Testing Dataset
                    - Import New Testing Dataset from File
      [2] Network Training submenu
            * Current Training Status: 'Trained' || 'Untrained'
            * Re-/Train Network (Using Current Training Config)         **BASED ON TRAINING STATUS**
            * Current Training Configuration
                + Training Dataset
                + Epochs
                + Batch Size
                + Learning Rate
            * Adjust Training Configuration                             **INFORM THAT ANY ADJUSTMENT CHANGES STATUS TO UNTRAINED**
                + Change Training Settings
                    - Epochs
                    - Batch Size
                    - Learning Rate
                + Change Training Dataset
                    - Previously Saved Training Dataset
                    - New Training Dataset From File                               
      [3] Network Inference submenu                                   **COMMENT OUT UNTIL FEATURE IS IMPLEMENTED SO LINE IS NOT INCLUDED IN MENU STRING**
            * ...
            * ...
      [4] Network Weights & Biases submenu
            * Import                                                    **NEED TO VERIFY MATCH WITH NETWORK SPECS**
            * Export
                + CSV
                + (eventually other formats will be added) 
            * Randomize Weights & Biases                                **INFORM THAT RANDOMIZING WILL CHANGE STATUS TO UNTRAINED**
      [5] Save Network State submenu

      [9] Return to 'Network Selection' menu
      [0] Exit Interface
     */
    private static int networkMenu(Console console) {
        int select = -1;
        
        String netMenu = "\t\t - MNIST Digit Classifier -\n\nNetwork Menu\n";
        netMenu += "\t[3] : Display network accuracy on MNIST training data\n";
        netMenu += "\t[4] : Display network accuracy on MNIST testing data\n";
        netMenu += "\t[5] : Run network on MNIST testing data (Images/Labels for all inputs)\n";
        netMenu += "\t[6] : Run network on MNIST testing data (Images/Labels for misclassified inputs)\n";
        netMenu += "\t[7] : Save current network state\n";
        netMenu += "\t[8] : Export network weights & biases to CSV\n\n";
        netMenu += "\t[9] : Return to 'Network Selection' menu\n";
        netMenu += "\t[0] : Exit Interface\n\n";
        netMenu += "Enter Selection: ";
        
        /* TODO: Not finished */

        return select;
    }

    private static int netSelectMenu(Console console) {
        String selectMenu = "\t\t - MNIST Digit Classifier -\n\nNetwork Selection Menu\n";
        selectMenu += "\t[1] : Create new network to train\n";
        selectMenu += "\t[2] : Load previously saved network state\n\n";
        selectMenu += "\t[0] : Exit interface\n\nEnter selection: ";
        
        return promptUserForSelection(console, selectMenu, new int[] {0, 1, 2});
    }

    private static int newNetworkMenu(Console console) {
        String newNetMenu = "\t\t -+-  Neural Network  -+-\n\nNetwork Creation Menu\n";
        newNetMenu += "\t[1] : Create new default MNIST digit classify network\n";
        newNetMenu += "\t[2] : Create new custom neural network\n\n";
        newNetMenu += "\t[9] : Return to 'Network Selection' menu\n";
        newNetMenu += "\t[0] : Exit interface\n\n";
        newNetMenu += "Enter selection: ";

        return promptUserForSelection(console, newNetMenu, new int[] {0, 9, 1, 2});
    }

    private static int loadNetworkMenu(Console console, NetworkContainer neuralNet) {
        int select = -1;

        String savedNetworksStr = "";

        String loadNetMenu = "\t\t -+-  Neural Network  -+-\n\nLoad Network Menu\n";
        loadNetMenu += "\t[1] : Load previously saved network by name\n";
        loadNetMenu += "\t[2] : List names of previously saved networks\n\n";
        loadNetMenu += "\t[9] : Return to 'Network Selection' menu\n";
        loadNetMenu += "\t[0] : Exit interface\n\n";
        loadNetMenu += "Enter selection: ";
        
        while(true) {
            select = promptUserForSelection(console, loadNetMenu, new int[] {0, 9, 1, 2});
            if(select == 0 || select == 9) return select;
            
            if(select == 1) {
                String inputNetName = "\t\t -+-  Neural Network  -+-\n\nEnter name of previously saved network (case-sensitive)...\n";
                inputNetName += "Enter name: ";

                String in, netFilenameStr, selectedNet;
                in = "";
                netFilenameStr = "%s.nn";

                try {
                    clearTerminal(console);
                    in = console.readLine(inputNetName);
                    selectedNet = String.format(netFilenameStr, in);
                    
                    if(!networkFilenameIsValidSavedNetwork(selectedNet)) throw new FileNotFoundException();
                    
                    if(neuralNet.loadNetwork(selectedNet)) return 1;
                    else throw new UnknownError("failed to load network from valid filename");

                } catch(FileNotFoundException e) {
                    clearTerminal(console);
                    console.readLine(
                            String.format("Invalid Network Name: No saved network with name '%s' exists!\n\nPress 'Enter' to return to previous menu...\n", in));
                    select = -1;
                }
            } else if(select == 2) {
                if(savedNetworksStr.isEmpty() || savedNetworksStr == null) savedNetworksStr = getSavedNetworksStr();
                displayInfoToUser(console, savedNetworksStr);
                select = -1;
            }
        }
    }

    private static int selectDatasetsMenu(Console console, NetworkContainer neuralNet) {
        // returns:  1 -> Both DataSet were selected,  2 -> Return to select datasets menu,  0 -> Exit,  9 -> Network select,  -1 -> Error
        int select = -1;

        String selectDataMenu = "\t\t -+-  Neural Network  -+-\n\nSelect Datasets Menu\n";
        selectDataMenu += "\t[1] : Select default MNIST training/testing datasets\n";
        selectDataMenu += "\t[2] : Import new training/testing datasets from disk\n\n";
        selectDataMenu += "\t[9] : Return to 'Network Selection' menu\n";
        selectDataMenu += "\t[0] : Exit interface\n\n";
        selectDataMenu += "Enter selection: ";

        select = promptUserForSelection(console, selectDataMenu, new int[] {0, 9, 1, 2});
        if(select == 0 || select == 9) return select;
        
        if(select == 1) {
            if(neuralNet.network().inputSize() == 784 && neuralNet.network().outputSize() == 10) {
                neuralNet.setTrainingData(DataSet.defaultMnistTrainingSet());
                neuralNet.setTestingData(DataSet.defaultMnistTestingSet());
                return 1;
            } else {
                String badConfigPrompt = "\t\t -+-  Neural Network  -+-\n\nDefault MNIST Datasets Selected\n\n";
                badConfigPrompt += "Current Configuration Incompatible with Default MNIST Datasets\n\n";

                int[] configs = {0, 0};     // if incompatible, configs[i]=1 (where (i=0) => input size  &&  (i=1) => output size)
                String currConfiguration = "";
                if(neuralNet.network().inputSize() != 784) {
                    currConfiguration += "\t  Default MNIST Data Input Size:    784\n";
                    currConfiguration += String.format("\t  Currently Configured Input Size:  %d\n\n", neuralNet.network().inputSize());
                    configs[0] = 1;
                }
                
                if(neuralNet.network().outputSize() != 10) {
                    currConfiguration += "\t  Default MNIST Data Output Size:    10\n";
                    currConfiguration += String.format("\t  Currently Configured Output Size:  %d\n\n", neuralNet.network().outputSize());
                    configs[1] = 1;
                }

                badConfigPrompt += currConfiguration;                

                String badConfig;
                if(configs[0] == 1 && configs[1] == 1) badConfig = "input and output sizes";
                else badConfig = (configs[0] == 1) ? "input size" : "output size";
                
                badConfigPrompt = String.format("The currently configured %s must be adjusted to match the chosen data...\n", badConfig);
                badConfigPrompt += String.format("\t[1] : Re-configure %s to match default MNIST data\n", badConfig);
                badConfigPrompt += "\t[2] : Keep current configuration and return to 'Select Datasets Menu'\n\n";
                badConfigPrompt += "\t[9] : Return to 'Network Selection' menu\n";
                badConfigPrompt += "\t[0] : Exit interface\n\n";
                badConfigPrompt += "Enter selection: ";

                select = promptUserForSelection(console, badConfigPrompt, new int[] {0, 9, 1, 2});
                if(select == 0 || select == 9 || select == 2) return select;

                if(select == 1) {
                    if(configs[0] > 0) neuralNet.network().adjustInputSize(784);

                    if(configs[1] > 0) neuralNet.network().adjustOutputSize(10);

                    neuralNet.setTrainingData(DataSet.defaultMnistTrainingSet());
                    neuralNet.setTestingData(DataSet.defaultMnistTestingSet());
                    return 1;
                } 
            
                return -1;
            }
        } else if(select == 2) {
            String trainFilename, testFilename, setTypeImportPrompt, inputFilePathPrompt, pathStr, filename, ext;
            trainFilename = testFilename = "NOT Selected...";

            String setTypeImportStr = "\t\t -+-  Neural Network  -+-\n\nImport Datasets Menu\n";
            setTypeImportStr += "\t[1] : Import training dataset\n";
            setTypeImportStr += "\t      Training dataset selected:  %s\n";
            setTypeImportStr += "\t[2] : Import testing dataset\n";
            setTypeImportStr += "\t      Testing dataset selected:   %s\n\n";
            setTypeImportStr += "\t[9] : Return to 'Network Selection' menu\n";
            setTypeImportStr += "\t[0] : Exit interface\n\n";
            setTypeImportStr += "Enter selection: ";

            String inputFilePathStr =  "\t\t -+-  Neural Network  -+-\n\nImport Dataset From File\n\n";
            inputFilePathStr += "Please input the desired %s filename as an ABSOLUTE path...\n";
            inputFilePathStr += "Enter filename: ";

            Path path;
            boolean trainIsSet, testIsSet;
            trainIsSet = testIsSet = false;
            while(true) {
                try {
                    setTypeImportPrompt = String.format(setTypeImportStr, trainFilename, testFilename);
                    select = promptUserForSelection(console, setTypeImportPrompt, new int[] {0, 9, 1, 2});
                    if(select == 0 || select == 9) return select;

                    /* TODO: add menu asks whether to import prev saved data from `resources/data/user/` or input an absolute path to new data and copy it to `resources/data/user/` */
                    /* TODO: then loop: menu asking for unique name to assign to dataset, prompts options to use filename or input new unique name, verify no file with name exists in  
                            `resources/data/user/` and copy file as name (appended with OG file extension) if no file exists; otherwise restart loop */
                    /* TODO: dataset file should always be present in `resources/data/user` and then the relative path to file can be passed to create the new DataSet */

                    inputFilePathPrompt = String.format(
                                                    inputFilePathStr,
                                                    (select == 1) ? "training" : "testing");
                    
                    pathStr = console.readLine(inputFilePathPrompt);
                    path = Paths.get(pathStr);
                    if(Files.exists(path) && !Files.isDirectory(path)) {
                        filename = path.getFileName().toString();
                        DataSet dataset = DataSet.newDataSetFromFile(
                                                                filename, 
                                                                (select == 1) ? DataSetType.TRAIN : DataSetType.TEST, 
                                                                neuralNet.network().outputSize(), 
                                                                pathStr);

                        if(select == 1) {
                            neuralNet.setTrainingData(dataset);
                            trainFilename = filename;
                            trainIsSet = true;
                        } else if(select == 2) {
                            neuralNet.setTestingData(dataset);
                            testFilename = filename;
                            testIsSet = true;
                        } else throw new Exception();

                        if(trainIsSet && testIsSet) return 1;
                    } else {
                        clearTerminal(console);
                        console.readLine(
                                    String.format("\n\nFile Not Found: The file '%s' is invalid or does not exist...\n\nPlease verify the path where the file exists and try again.\n\nPress 'Enter' key to return to previous menu...\n", pathStr));
                        select = -1;
                    }
                /* TODO: Handle the custom exceptions thrown by DataSet once custom exceptions are implemented */
                } catch(Exception e) {
                    /* TODO: Handle the exception thrown */
                }
            }
        }
        
        return -1;
    }

    private static int newCustomNetMenu(Console console, NetworkContainer neuralNet) {
        int select = -1;
        String in, currStr, missing, netName;
        in = currStr = missing = netName = "";

        ConfigInput result;
        String inSizeStr, outSizeStr, hidLayersStr, hidSizesStr, epochsStr, batchStr, learnStr;
        int inSize, outSize, hidLayers, epochs, batch, curr;
        
        inSizeStr = outSizeStr = hidLayersStr = hidSizesStr = epochsStr = batchStr = learnStr = "NOT configured...";
        inSize = outSize = hidLayers = epochs = batch = curr = -1;
        int[] hidSizes = {};
        double learn = -1.f;

        boolean[] validParams = new boolean[7];
        for(int i=0; i<validParams.length; i++) validParams[i] = false;
        String[] paramNames = {
                "Input layer size",
                "Output layer size",
                "Total hidden layers",
                "Hidden layer sizes",
                "Training batch size",
                "Training epochs",
                "Learning rate"};
        
        boolean configConfirmed = false;

        String customNetMenu = "\t\t -+-  Neural Network  -+-\n\nConfigure Network Menu\n\n";
        customNetMenu += "Network Configuration\n";
        customNetMenu += "\t[1] : Change input layer size\n";
        customNetMenu += "\t      Input layer size:      %s\n";
        customNetMenu += "\t[2] : Change output layer size\n";
        customNetMenu += "\t      Output layer size:     %s\n";
        customNetMenu += "\t[3] : Change total hidden layers\n";
        customNetMenu += "\t      Total hidden layers:   %s\n";
        customNetMenu += "\t[4] : Change hidden layer sizes\n";
        customNetMenu += "\t      Hidden layer sizes:    %s\n";
        customNetMenu += "\t[5] : Change training batch size\n";
        customNetMenu += "\t      Training batch size:   %s\n";
        customNetMenu += "\t[6] : Change training epochs\n";
        customNetMenu += "\t      Training epochs:       %s\n";
        customNetMenu += "\t[7] : Change learning rate\n";
        customNetMenu += "\t      Learning rate:         %s\n\n";
        customNetMenu += "\t[8] : Confirm network configuration\n\n";
        customNetMenu += "\t[9] : Return to 'Network Selection' menu\n";
        customNetMenu += "\t[0] : Exit interface\n\n";
        customNetMenu += "Enter selection: ";

        String hidSizesPrompt = "Configuring size for Hidden Layer %d of %d...\n\nPlease input the valid positive integer for the desired size of this hidden layer.\n\nEnter size of Hidden Layer %d: ";

        while(true) {
            try {
                while(!configConfirmed) {
                    clearTerminal();
                    String currMenu = String.format(customNetMenu, 
                                                    inSizeStr,
                                                    outSizeStr,
                                                    hidLayersStr,
                                                    hidSizesStr,
                                                    batchStr,
                                                    epochsStr,
                                                    learnStr);
                    
                    in = console.readLine(currMenu);
                    select = Integer.parseInt(in);

                    if(select == 0 || select == 9) return select;
                    else if(select == 1) {
                        result = getIntConfigInput(console, "Input Layer Size...\n\nEnter valid positive integer value: ", 0, Integer.MAX_VALUE);
                        if(result == null) return -1;
                        else if(result.input() == 9) return result.input();
                        else {
                            inSizeStr = result.inputStr();
                            inSize = result.input();
                            validParams[0] = true;
                        }
                    } else if(select == 2) {
                        result = getIntConfigInput(console, "Output Layer Size...\n\nEnter valid positive integer value: ", 0, Integer.MAX_VALUE);
                        if(result == null) return -1;
                        else if(result.input() == 9) return result.input();
                        else {
                            outSizeStr = result.inputStr();
                            outSize = result.input();
                            validParams[1] = true;
                        }
                    } else if(select == 3) {
                        if(validParams[3]) {
                            while(true) {
                                clearTerminal();
                                currStr = "\n\nWARN: Adjusting the total number of hidden layers will invalidate the currently configured hidden layer sizes!\n\n";
                                currStr += "[1] : Change number of hidden layers & reconfigure hidden layer sizes\n";
                                currStr += "[2] : Keep current number of hidden layers & return to previous menu\n\n";
                                currStr += "Enter selection: ";

                                int k;
                                String response = console.readLine(currStr);
                                if((k = Integer.parseInt(response)) == 1) {
                                    int[] clearSizes = {};
                                    hidSizes = clearSizes;
                                    hidSizesStr = "NOT configured...";
                                    validParams[3] = false;
                                    break;
                                } else if(k == 2) break;
                                else console.readLine(String.format(
                                        "\n\nInvalid input: Failed to parse user response '%d'...\nPlease ensure your responses are valid values and then try again.\n\nPress 'Enter' to return to previous menu...\n", k));
                            }
                        }
                        if(!validParams[3]) {
                            result = getIntConfigInput(console, "Total Number of Hidden Layers...\n\nEnter valid positive integer value: ", 0, 25);
                            if(result == null) return -1;
                            else if(result.input() == 9) return result.input();
                            else {
                                hidLayersStr = result.inputStr();
                                hidLayers = result.input();
                                validParams[2] = true;
                            }
                        }
                    } else if(select == 4) {
                        if(validParams[2] && hidLayers > 0) {
                            clearTerminal();
                            hidSizes = new int[hidLayers];
                            hidSizesStr = "[ ";
                            for(int i=0; i<hidSizes.length; i++) {
                                while(curr < 1) {
                                    currStr = console.readLine(String.format(hidSizesPrompt, i+1, hidLayers, i+1));
                                    curr = Integer.parseInt(currStr);
                                    if(curr > 0) break;
                                    else {
                                        clearTerminal();
                                        console.readLine(String.format("\n\nInvalid layer size: A layer cannot be configured with a size of %d!\nPlease use valid positive integers to indicate layer sizes.\n\nPress 'Enter' to return to previous menu...\n", curr));
                                    }
                                }
                                
                                hidSizes[i] = curr; 
                                hidSizesStr += (i == hidLayers-1) ? String.format("%d, ", hidSizes[i]) : String.format("%d ]", hidSizes[i]);
                                curr = -1;
                            }
                            validParams[3] = true;
                        } else if(hidLayers == 0) {
                            if(hidSizes.length > 0) {
                                int[] zeroSizes = {};
                                hidSizes = zeroSizes;
                            }
                            hidSizesStr = "[ NONE ]";
                            validParams[3] = true;
                        } else {
                            clearTerminal();
                            console.readLine("Cannot configure hidden layer sizes until the total number of hidden layers has been set!\nPlease indicate the total number of hidden layers and then configure their sizes here.\n\nPress 'Enter' to return to previous menu...\n");
                        }
                    } else if(select == 5) {
                        result = getIntConfigInput(console, "Training Batch Size...\n\nEnter valid positive integer value: ", 0, 200);
                        if(result == null) return -1;
                        else if(result.input() == 9) return result.input();
                        else {
                            batchStr = result.inputStr();
                            batch = result.input();
                            validParams[4] = true;
                        }
                    } else if(select == 6) {
                        result = getIntConfigInput(console, "Total Training Epochs...\n\nEnter valid positive integer value: ", 0, 100000);
                        if(result == null) return -1;
                        else if(result.input() == 9) return result.input();
                        else {
                            epochsStr = result.inputStr();
                            epochs = result.input();
                            validParams[5] = true;
                        }
                    } else if(select == 7) {
                        while(true) {
                            try {
                                clearTerminal();
                                currStr = console.readLine("Learning Rate...\n\tThe learning rate 'eta' must be a float between 0.0 - 1.0 (exclusive), i.e. 0.0 < 'eta' < 1.0\n\nEnter valid floating point value: ");
                                double eta = Double.parseDouble(currStr);
                                if(eta > 0.f && eta < 1.f) {
                                    learnStr = currStr;
                                    learn = eta;
                                    validParams[6] = true;
                                    break;
                                } else throw new NumberFormatException();
                            } catch(NumberFormatException e) {
                                clearTerminal();
                                currStr = console.readLine(
                                        "\n\nInvalid input: value must be float between 0.0 - 1.0 (exclusive)\n\n\t[9] : Return to `Network Selection` menu\n\t\t  ANY other key to re-enter a valid value\n\nEnter selection: ");
                                int retrySelect = Integer.parseInt(currStr);
                                if(retrySelect == 9) return retrySelect;
                            }
                        }
                    } else if(select == 8) {
                        missing = "";
                        boolean paramsAreSet = true;
                        for(int i=0; i<validParams.length; i++) {
                            if(!validParams[i]) {
                                missing += (paramsAreSet) ? String.format("\t%s", paramNames[i]) : String.format(",\n\t%s", paramNames[i]);
                                if(paramsAreSet) paramsAreSet = false;
                            }
                        }

                        if(paramsAreSet) {
                            currStr = learnStr.strip();

                            String confirmConfigStr = "\t\t -+-  New Neural Network  -+-\n\n";
                            confirmConfigStr += "Current Configuration\n";
                            confirmConfigStr += "\t  Input layer size:      %d\n";
                            confirmConfigStr += "\t  Output layer size:     %d\n";
                            confirmConfigStr += "\t  Total hidden layers:   %d\n";
                            confirmConfigStr += "\t  Hidden layer sizes:    %s\n";
                            confirmConfigStr += "\t  Training batch size:   %d\n";
                            confirmConfigStr += "\t  Training epochs:       %d\n";
                            confirmConfigStr += "\t  Learning rate:         %.";

                            // format string so float is formatted/rounded to same number of dec places as user input
                            confirmConfigStr += String.format("%df\n\n", currStr.substring(currStr.lastIndexOf('.')+1).length());
                            confirmConfigStr += "\t[1] : Create network with this configuration\n";
                            confirmConfigStr += "\t[2] : Return to previous `Network Configuration Menu`\n\n";
                            confirmConfigStr += "Enter selection: ";
                            
                            currStr = String.format(
                                                confirmConfigStr,
                                                inSize,
                                                outSize,
                                                hidLayers,
                                                Arrays.toString(hidSizes),
                                                batch,
                                                epochs,
                                                learn);
                                
                            int response = confirmConfiguration(console, confirmConfigStr);
                            if(response == 1) {
                                clearTerminal();
                                String netNamePrompt = "\t\t -+-  New Neural Network  -+-\n\n";
                                netNamePrompt += "Choose a name for the newly created network...\n";
                                netNamePrompt += "\t Valid characters include: [ A-Z | a-z | 0-9 | _ | - ]\n\n";
                                netNamePrompt += "Enter network name: ";
                                /* TODO: verify name consists only of valid characters */
                                netName = console.readLine(netNamePrompt);
                                /* TODO: verify no network with same name already exists on disk; prompt whether user wants to replace saved network with this newly created network if name already in use */
                                configConfirmed = true;
                                break;
                            }
                        } else {
                            clearTerminal();
                            missing += "\n\n";
                            String miss = "Failed to create custom network!\nThere are missing network parameters that are not configured...\n\nMissing Network Parameters:\n%s\n\n";
                            console.printf(miss, missing);
                            console.readLine("Please configure the missing network parameters and try again.\n\nPress 'Enter' to return to previous menu...\n");
                        }
                    }
                }
                if(!configConfirmed) return -1;
                neuralNet.setNetwork(new NeuralNetwork(
                                                netName,
                                                new NetworkConfig(
                                                            inSize,
                                                            outSize,
                                                            hidSizes,
                                                            batch,
                                                            epochs,
                                                            learn)));
                
                return 1;
                /* prompt user for filepaths for test/training data, create DataSets after read, assign the sets to container + return 99 */
            } catch(NumberFormatException e) {
                clearTerminal();
                console.readLine(
                        String.format("\n\nInvalid input: Failed to parse user response '%s'...\nPlease ensure your responses are valid and then try again.\n\nPress 'Enter' to return to previous menu...\n", in));
            }
        }    
    }

    // Prompts user for input and returns record containing str input + parsed int if parsed int >= `minVal` and <= `maxVal`
    private static ConfigInput getIntConfigInput(Console console, String prompt, int minVal, int maxVal) {
        String in = "";
        int select = -1;

        while(true) {
            clearTerminal();
            try {
                in = console.readLine(prompt);
                select = Integer.parseInt(in);
                if(select >= minVal && select <= maxVal) return new ConfigInput(in, select);
            } catch(NumberFormatException e) {
                clearTerminal();
                String retryIn = console.readLine(String.format(
                        "\n\nInvalid input: value must be an integer between %d and %d (inclusive)\n\n\t[9] : Return to `Network Selection` menu\n\t\t  ANY other key to re-enter a valid value\n\nEnter selection: ",
                        minVal, maxVal));
                int retrySelect = Integer.parseInt(retryIn);
                if(retrySelect == 9) return new ConfigInput(retryIn, retrySelect);
            }
        } 
    }

    private static NetworkContainer initDefaultNetwork() {
        return NeuralNetwork.defaultMnistNetwork();
    }

    private static void clearTerminal(){
        System.out.print("\033[H");     // move cursor to top left of terminal 
        System.out.print("\033[2J");    // clear text after cursor
        System.out.flush();
    }

    private static void clearTerminal(Console console) {
        console.writer().print("\033[H");         // move cursor to top left of terminal
        console.writer().print("\033[2J");        // clear all text after cursor
        console.writer().flush();
    }

    private static void updateSavedNetworksList() {
        savedNetworks = null;

        String pathStr = "src/resources/networks";
        Path networkDir = Paths.get(pathStr).toAbsolutePath().normalize();      // Need to look at docs to see if abs/normalize steps are necessary
        
        try (Stream<Path> stream = Files.walk(networkDir)) {
            savedNetworks = stream.filter(Files::isRegularFile)
                                    .filter(path -> path.getFileName().toString().endsWith(".nn"))
                                    .collect(Collectors.toList());

            if(savedNetworks == null) throw new UnknownError("list of saved networks updated but still null");
        } catch (IOException e) {
            e.printStackTrace();
            throw new UnknownError("failed to update list of saved networks");
        }
    }

    private static String getSavedNetworksStr() {
        if(savedNetworks.isEmpty() || savedNetworks == null) updateSavedNetworksList();

        String networksListStr = "\t\t -+-  Neural Network  -+-\n\nSaved Networks...\n";
        String netStr = "\t %s \n";

        int dotIdx;
        String currStr;
        for(Path network : savedNetworks) {
            currStr = network.getFileName().toString();
            if(currStr.isEmpty() || currStr == null) continue;

            dotIdx = currStr.lastIndexOf(".nn");
            if(dotIdx < 0) throw new UnknownError(String.format("failed to process network filename: '%s'", currStr));

            networksListStr += String.format(netStr, currStr.substring(0, dotIdx));
        }

        return networksListStr;
    }

    private static void displayInfoToUser(Console console, String info) {
        String infoPrompt = String.format("%s", info);

        infoPrompt += "\nPress 'Enter' to return to previous menu...\n";

        clearTerminal(console);
        console.readLine(infoPrompt);
    }

    private static boolean networkFilenameIsValidSavedNetwork(String filename) {
        if(savedNetworks.isEmpty() || savedNetworks == null) updateSavedNetworksList();
        return savedNetworks.stream()
                            .map(Path::getFileName)
                            .filter(java.util.Objects::nonNull)
                            .anyMatch(file -> file.toString().equals(filename));
    }

    private static int confirmConfiguration(Console console, String confirmPrompt) {
        String in = "";
        int select = -1;

        while(true) {
            try {
                clearTerminal();
                in = console.readLine(confirmPrompt);
                if((select = Integer.parseInt(in)) == 1 || select == 2) return select;
                else throw new NumberFormatException();
            } catch(NumberFormatException e) {
                clearTerminal();
                console.readLine(
                        String.format("\n\nInvalid input: '%s' is not a valid selection...\n\nPress 'Enter' to return and change your selection...\n", in));
                select = -1;
            }
        }
    }

    private static int promptUserForSelection(Console console, String prompt, int[] validOptions) {
        String in = "";
        int select = -1;

        while(true) {
            try {
                clearTerminal();
                in = console.readLine(prompt);
                select = Integer.parseInt(in);
                for(int i=0; i<validOptions.length; i++) {
                    if(select == validOptions[i]) return select;
                }
                throw new NumberFormatException();
            } catch(NumberFormatException e) {
                clearTerminal();
                console.readLine(
                        String.format("\n\nInvalid input: '%s' is not a valid selection...\n\nPress 'Enter' to return to previous menu and change your selection...\n", in));
                select = -1;
            }
        }
    }


    private record ConfigInput(String inputStr, int input) {}


  /*
   // Interface shown once a network is ready for use (trained or imported) allowing... 
   //    - display of network accuracy for both mnist training and testing datasets
   //    - run network on mnist test data showing images/labels for all test inputs
   //    - run network on mnist test data showing images/labels only for misclassified inputs
   //    - save the current configuration of the network
   //    - return to network select menu
   //    - exit
   //  
  private static void startNetworkMenu(){
    boolean goMainMenu = false;  // flag signaling to return to the network select menu
    boolean exportSuccess = false;  // flag signaling that the current network was successfully written to disk
  

    //     - MNIST Digit Classifier -
    // Current Network Menu
    //   [3] : Display Network Accuracy on MNIST Training Data
    //   [4] : Display Network Accuracy on MNIST Testing Data
    //   [5] : Run Network on MNIST Testing Data (Images/Labels for All Inputs)
    //   [6] : Run Network on MNIST Testing Data (Images/Labels for Misclassified Inputs)
    //   [7] : Save Current Network Configuration
    //
    //   [9] : Return to Network Select Menu  
    //   [0] : Exit Interface
    //
    // Enter Selection: 
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
  */    
}

