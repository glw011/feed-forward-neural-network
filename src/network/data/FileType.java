package src.network.data;

public enum FileType {
    CSV(".csv"),
    JSON(".json"),
    UNK("UNKNOWN");

    private final String ext;

    private FileType(String ext) {
        this.ext = ext;
    }

    public String ext() {return this.ext;}
}
