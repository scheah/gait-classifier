import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.FileReader;

public class main {

    public static void addFilesFromDir(GaitClassifier classifier, String directoryPath, String className) {
        System.out.println("Reading data of class " + className + " from directory " + directoryPath);
        try {
            File folder = new File(directoryPath);
            BufferedReader reader = null;
            File [] listOfFiles = folder.listFiles();
            for (int i = 0; i < listOfFiles.length; i++) {
                reader = new BufferedReader(new FileReader(listOfFiles[i]));
                classifier.loadGaitData(reader, className);
                reader.close();
            }
        }
        catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public static void main(String[] args) {
        GaitClassifier m_classifier = new GaitClassifier(new String[]{"brush_teeth", "climb_stairs", "comb_hair",
            "descend_stairs", "drink_glass", "eat_meat", "eat_soup", "getup_bed", "liedown_bed", "pour_water", "sitdown_chair",
            "standup_chair", "use_telephone", "walk"});
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Brush_teeth", "brush_teeth");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Climb_stairs", "climb_stairs");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Comb_hair", "comb_hair");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Descend_stairs", "descend_stairs");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Drink_glass", "drink_glass");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Eat_meat", "eat_meat");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Eat_soup", "eat_soup");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Getup_bed", "getup_bed");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Liedown_bed", "liedown_bed");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Pour_water", "pour_water");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Sitdown_chair", "sitdown_chair");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Standup_chair", "standup_chair");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Use_telephone", "use_telephone");
        addFilesFromDir(m_classifier, "dataset/HMP_Dataset/Walk", "walk");
        m_classifier.train();
        System.out.println(m_classifier.dataSummary());
        System.out.println(m_classifier.internalTestSummary(5));
        // System.out.println(m_classifier.dataDump());
    }
}
