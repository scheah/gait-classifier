import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.FileReader;
import java.util.Scanner;

public class main {

    public static void addFilesFromDir(GaitClassifier classifier, String directoryPath, String className) {
        System.out.println("Reading data of class " + className + " from directory " + directoryPath);
        try {
            File folder = new File(directoryPath);
            BufferedReader reader = null;
            File [] listOfFiles = folder.listFiles();
            for (int i = 0; i < listOfFiles.length; i++) {
                if (listOfFiles[i].toString().contains("Gyro") || listOfFiles[i].toString().contains("Accelerometer")) {
                    reader = new BufferedReader(new FileReader(listOfFiles[i]));
                    classifier.loadGaitData(reader, className);
                    reader.close();
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
        }
    }

    public static void processHMPData(int classOption, int featureOption) {
        String [] actionClasses = new String[]{"brush_teeth", "climb_stairs", "comb_hair",
            "descend_stairs", "drink_glass", "eat_meat", "eat_soup", "getup_bed", "liedown_bed", "pour_water", "sitdown_chair",
            "standup_chair", "use_telephone", "walk"};
        String [] userClasses = new String[]{"f1", "f2", "f3", "f4", "f5", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9",
            "m10", "m11"};
        GaitClassifier m_classifier = null;
        if (classOption == 1) {
            m_classifier = new GaitClassifier(actionClasses, featureOption);
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Brush_teeth", "brush_teeth");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Climb_stairs", "climb_stairs");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Comb_hair", "comb_hair");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Descend_stairs", "descend_stairs");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Drink_glass", "drink_glass");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Eat_meat", "eat_meat");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Eat_soup", "eat_soup");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Getup_bed", "getup_bed");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Liedown_bed", "liedown_bed");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Pour_water", "pour_water");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Sitdown_chair", "sitdown_chair");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Standup_chair", "standup_chair");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Use_telephone", "use_telephone");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_action/Walk", "walk");
        }
        else if (classOption == 2) {
            m_classifier = new GaitClassifier(userClasses, featureOption);
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/f1", "f1");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/f2", "f2");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/f3", "f3");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/f4", "f4");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/f5", "f5");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m1", "m1");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m2", "m2");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m3", "m3");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m4", "m4");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m5", "m5");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m6", "m6");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m7", "m7");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m8", "m8");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m9", "m9");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m10", "m10");
            addFilesFromDir(m_classifier, "dataset/HMP_Dataset/by_user/m11", "m11");
        }
        else {
            System.out.println("Invalid classification choice");
            return;
        }
        m_classifier.train();
        System.out.println(m_classifier.dataSummary());
        System.out.println(m_classifier.internalTestSummary(5));
        // System.out.println(m_classifier.dataDump());
        // try {
        //     m_classifier.saveArff("HMP_Dataset");
        // }
        // catch (Exception e) {
        //     e.printStackTrace();
        // }
    }

	public static void processLGWatchUrbaneData(int classOption, int featureOption) {
        String [] actionClasses = new String[]{"cup", "door", "typing", "walking", "watch"};
        String [] userClasses = new String[]{"andrew", "derek", "scott", "sebastian"};
        GaitClassifier m_classifier = null;
        if (classOption == 1) {
            m_classifier = new GaitClassifier(actionClasses, featureOption);
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_action/Cup", "cup");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_action/Door", "door");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_action/Typing", "typing");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_action/Walking", "walking");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_action/Watch", "watch");
        }
        else if (classOption == 2) {
            m_classifier = new GaitClassifier(userClasses, featureOption);
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_user/Andrew", "andrew");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_user/Derek", "derek");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_user/Scott", "scott");
            addFilesFromDir(m_classifier, "dataset/LG_Watch_Urbane/by_user/Sebastian", "sebastian");
        }
        else {
            System.out.println("Invalid classification choice");
            return;
        }
        m_classifier.train();
        System.out.println(m_classifier.dataSummary());
        System.out.println(m_classifier.internalTestSummary(5));
        // System.out.println(m_classifier.dataDump());
        // try {
        //     m_classifier.saveArff("LG_Watch_Urbane");
        // }
        // catch (Exception e) {
        //     e.printStackTrace();
        // }
	}

    public static void processDataset(int dataOption, int classOption, int featureOption) {
        switch(dataOption) {
            case 1:
                processHMPData(classOption, featureOption);
                break;
            case 2:
                processLGWatchUrbaneData(classOption, featureOption);
                break;
            default:
                System.out.println("Invalid dataset choice");
        }
    }

    public static void main(String[] args) {
        Scanner reader = new Scanner(System.in);
        System.out.println("Specify which data you would like to process: ");
        System.out.println("\t1: HMP_Dataset\n\t2: LG_Watch_Urbane");
        int dataOption = reader.nextInt();
        System.out.println("Classify by action or user?: ");
        System.out.println("\t1: Action\n\t2: User");
        int classOption = reader.nextInt();
        System.out.println("Extract features using: ");
        System.out.println("\t1: Gait\n\t2: General time domain");
        int featureOption = reader.nextInt();
        processDataset(dataOption, classOption, featureOption);
    }
}
