The authors of **India Driving Dataset (IDD): A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments** highlight a notable gap in existing datasets, which primarily focus on structured driving environments with well-defined infrastructure, limited traffic categories, and adherence to traffic rules. To fill this void, the authors present IDD, a novel dataset tailored for road scene understanding in unstructured environments, specifically on Indian roads. The updated version of the dataset <i>(acquired in Oct, 2023)</i> comprises 20k images, meticulously annotated with 41 classes, derived from 182 drive ***sequence***.

There are 2 different versions available at DatasetNinja:

* IDD: Segmentation (current)
* IDD: [Object Detection](https://datasetninja.com/idd-detection)

IDD diverges from popular benchmarks like Cityscapes, introducing an expanded label set to accommodate new classes and reflecting label distributions that deviate significantly from existing datasets. The dataset captures the complexity of unstructured road scenes, featuring classes with greater within-class diversity. Additionally, IDD identifies new classes such as drivable areas beyond the road. The authors propose a four-level label hierarchy (***level1id***, ***level2id***, ***level3id***,  ***level4id***) to allow varying levels of complexity, opening avenues for new training methods. <i>Please note, that some labels were not specified by their ***category***</i>

<img src="https://github.com/supervisely/dataset-tools/assets/78355358/d8b6b2bf-cb27-4077-86fe-7697911cf0b7" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Label distribution in the initial IDD-10k dataset. The following information is shown here: (i) pixel counts of individual labels on the y-axis (ii) four-level label hierarchy used by the dataset at the bottom, (iv) the color legend for the predicted and ground truth masks shown in the paper is used for the corresponding bars. There are 4 levels of the hierarchy giving different complexity levels for training models</span>


Empirical studies demonstrate that state-of-the-art methods for semantic segmentation exhibit lower accuracies on IDD compared to Cityscapes ([available on DatasetNinja](https://datasetninja.com/cityscapes)), emphasizing its distinct characteristics. The authors suggest that IDD presents an ideal opportunity for exploring new challenges such as domain adaptation, few-shot learning, and behavior prediction in road scenes.

Autonomous navigation is rapidly advancing, and the availability of large-scale datasets is a crucial contributor to this progress. However, challenges persist, particularly in achieving data scale and diversity necessary for ensuring safety and reliability in diverse and unstructured environments. The authors assert that IDD, with its focus on less structured environments, better aligns with the needs of autonomous navigation in various parts of the world, including Asia, South America, and Africa.

## Dataset properties

The dataset is collected in India, offering road scenes that differ significantly from those in Europe or North America. Indian roads exhibit a greater variety of traffic participants, including unique classes such as autorickshaws and animals. Within-class diversity is higher due to variations in vehicle manufacturing years and wear. The dataset's distribution of classes, even those overlapping with Cityscapes, differs significantly, underscoring the need for a more diverse and complex dataset.

IDD incorporates extensive variations in ambient conditions, such as lighting, shadows, clouded skies, and particulate matter, further increasing its complexity. The dataset is annotated at a scale significantly larger than contemporary datasets like KITTI or Cityscapes, providing high-quality annotations for 41 labels.

The authors conduct a thorough analysis of the dataset, pointing out challenges observed when applying models trained on existing datasets. They emphasize ambiguous road boundaries, diversity in vehicles and pedestrians, extensive use of information boards, and variations in ambient conditions as distinctive features of unstructured environments.

## Dataset acquisition, pre-processing and statistics

The dataset, acquired from Bangalore and Hyderabad cities in India, comprises 182 drive sequences with a mix of urban and rural areas, highways, single and double-lane roads. The driving conditions are highly unstructured due to the following reasons:
1. These cities are experiencing rapid growth, marked by extensive construction near roads. 
2. Additionally, road boundaries lack clear definition.
3. Pedestrians and jaywalkers are abundant in road images.
4. A high density of motorbikes and trucks on the roads. Moreover, there is considerable diversity in vehicle models.

Images are selected from forward-facing cameras, densely sampled around crowded and special interest places. The dataset is split into 70% train, 10% validation, and 20% test, with careful consideration for class imbalance.

<img src="https://github.com/supervisely/dataset-tools/assets/78355358/bc1cc12c-49e3-4e55-9a29-067fd0bbce52" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Comparison of the pixel count in our dataset with Cityscapes. The y axis is plotted in log-scale. Note that for most classes of vehicles, the number of pixels are 5-10 times more than Cityscapes. Moreover our dataset has newer labels like autorickshaw, billboard, drivable/nondrivable fallback which also have significant number of labeled pixels.</span>

Comparisons with other datasets, such as Cityscapes, KITTI, and Mapillary Vistas, highlight the unique characteristics of IDD. The authors present detailed label statistics, pixel counts, and instances of traffic participants, demonstrating the dataset's distinctions.

<img src="https://github.com/supervisely/dataset-tools/assets/78355358/e3f50948-ddd3-439a-b441-084af34c9256" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Comparison of traffic participants in our dataset with Cityscapes.</span>

In conclusion, IDD stands out as a comprehensive and challenging dataset for unstructured road scene understanding, offering new opportunities for advancing research in autonomous navigation and related domains.
