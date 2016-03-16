import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Path;
import java.util.Scanner;

public class main {

	private static final int HMP_DATASET = 1;
	private static final int LG_DATASET = 2;

	private static final int ACTION_CLASS = 1;
	private static final int USER_CLASS = 2;

	private static final int GAIT_CLASSIFIER = 1;
	private static final int TIME_DOMAIN_CLASSIFIER = 2;
	private static final int TIME_FREQ_DOMAIN_CLASSIFIER = 3;

	public static void main(String[] args) {
		Scanner reader = new Scanner(System.in);

		System.out.println("Specify which data you would like to process: ");
		System.out.println("\t1: HMP_Dataset\n\t2: LG_Watch_Urbane");
		int dataOption = reader.nextInt();

		System.out.println("Classify by action or user?: ");
		System.out.println("\t1: Action\n\t2: User");
		int classOption = reader.nextInt();

		System.out.println("Extract features using: ");
		System.out.println("\t1: Gait\n\t2: General time domain\n\t3: Time and freq domain");
		int featureOption = reader.nextInt();

		reader.close();

		processDataset(dataOption, classOption, featureOption);
	}

	private static void processDataset(int dataOption, int classOption, int featureOption) {
		DataClassifier classifier = null;
		String datasetDir = null;
		String classificationDir = null;
		String[] classes = null;

		switch (dataOption) {
		case HMP_DATASET:
			datasetDir = "dataset/HMP_Dataset/";
			switch (classOption) {
			case ACTION_CLASS:
				classificationDir = "by_action/";
				classes = new String[] { "brush_teeth", "climb_stairs", "comb_hair", "descend_stairs", "drink_glass",
						"eat_meat", "eat_soup", "getup_bed", "liedown_bed", "pour_water", "sitdown_chair",
						"standup_chair", "use_telephone", "walk" };
				break;
			case USER_CLASS:
				classificationDir = "by_user/";
				classes = new String[] { "f1", "f2", "f3", "f4", "f5", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8",
						"m9", "m10", "m11" };
				break;
			}
			break;
		case LG_DATASET:
			datasetDir = "dataset/LG_Watch_Urbane/";
			switch (classOption) {
			case ACTION_CLASS:
				classificationDir = "by_action/";
				classes = new String[] { "cup", "door", "typing", "walking", "watch" };
				break;
			case USER_CLASS:
				classificationDir = "by_user/";
				classes = new String[] { "andrew", "chris", "derrick", "scott", "sebastian", "matt", "justine", "jennifer", "jackie", "sabrina" };
				break;
			}
			break;
		}

		if (featureOption == GAIT_CLASSIFIER) {
			classifier = new GaitClassifier(classes);
		} else if (featureOption == TIME_DOMAIN_CLASSIFIER) {
			classifier = new TimeDomainClassifier(classes);
		} else if (featureOption == TIME_FREQ_DOMAIN_CLASSIFIER) {
			classifier = new TimeFreqDomainClassifier(classes);
		}

		for (String s : classes) {
			addFilesFromDir(classifier, datasetDir + classificationDir + s, s);
		}

		classifier.train();
		System.out.println(classifier.dataSummary());
		System.out.println(classifier.internalTestSummary(5));
	}

	private static void addFilesFromDir(DataClassifier classifier, String directoryPath, String className) {
		System.out.println("Reading data of class " + className + " from directory " + directoryPath);
		try {
			File[] listOfFiles = new File(directoryPath).listFiles();
			for (File file : listOfFiles) {
				Path path = file.toPath();
				if (path.getFileName().toString().startsWith("Gyro") || path.getFileName().toString().startsWith("Accelerometer")) {
					classifier.loadData(file, className);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println(e.getMessage());
		}
	}

}
