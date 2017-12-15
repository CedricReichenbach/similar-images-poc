package info.magnolia.ai.similarity;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

public class App {

    private static final String[] ALL_IMAGES = {
            "images/portraits/n_2017-09-30-05-45-31-1100x589.jpg",
            "images/portraits/n_young-woman-1119479_960_720.jpg",
            "images/portraits/n_girl-with-flowers-1374221_640.jpg",
            "images/portraits/n_LinusPaulingGraduation1922.jpg",
            "images/portraits/s_white-male-1754269_960_720.jpg",
            "images/portraits/n_man-1464787_960_720.jpg",
            "images/portraits/n_girl-1770829_960_720.jpg",
            "images/portraits/s_white-cute-red-child-clothing-christmas-baby-face-infant-toddler-head-skin-organ-adorable-blue-eyes-917499.jpg",
            "images/portraits/n_pexels-photo-691989.jpeg",
            "images/portraits/n_pexels-photo-247917.jpeg",
            "images/portraits/s_woman-in-santa-hat-1506441181V8C.jpg",
            "images/portraits/n_Male-Old-Poland-Dad-Fatigue-Man-Father-A-Person-114257.jpg",
            "images/portraits/n_India-Human-Hindu-Portrait-613601.jpg",
            "images/portraits/s_santa-958641_960_720.jpg",
            "images/portraits/s_santa-woman-portrait-1481305483lzf.jpg",
            "images/portraits/s_7457961848_f1ef1c3f0f_b.jpg",
            "images/portraits/n_pexels-photo-715822.jpeg",
            "images/portraits/s_woman-with-a-christmas-cracker-1480523778FaQ.jpg",
            "images/portraits/s_santa-woman-portrait-1481383803O3M.jpg",
            "images/portraits/n_male-2408532_960_720.jpg",
            "images/portraits/s_Snow-Fig-Owl-Contemplative-Christmas-Santa-Hat-1906639.jpg",
            "images/portraits/n_2017-04-11-18-49-49-725x485.jpg",
            "images/portraits/n_Hat-Cool-Fashion-Face-Portrait-Waif-996038.jpg",
            "images/portraits/n_Peter_Pace_official_portrait.jpg",
            "images/portraits/n_819px-Hillary_Clinton_official_Secretary_of_State_portrait_crop.jpg",
            "images/portraits/s_christmas-business-people-1478194721vrf.jpg",
            "images/portraits/n_Woman-Hat-Model-Beautiful-Portrait-Person-1845148.jpg",
            "images/portraits/s_A_Bigly_Christmas.jpg",
            "images/portraits/s_Santa-Hat-Red-Mackerel-Cat-Cute-Funny-Christmas-1898614.jpg",
            "images/portraits/n_Person-Man-Headshot-Hat-Male-Portrait-People-2334963.jpg",
            "images/portraits/n_woman-2999000_640.jpg",
            "images/mixed/US_Navy_091202-N-7280V-285_Sailors_dance_at_the_annual_Christmas_Disco_Party_for_the_disabled_during_a_community_outreach_project.jpg",
            "images/mixed/hands-437968_640.jpg",
            "images/mixed/kids-2989103_640.jpg",
            "images/mixed/cd-cover-2969102_640.jpg",
            "images/mixed/385933_10151318907492503_659232467_n.JPG",
            "images/mixed/'Mustangs'_host_Christmas_party,_build_family_morale_121211-A-CJ112-820.jpg",
            "images/mixed/1280px-Indoor_Hot_Tub_Christmas_Party.jpg",
            "images/mixed/121212-F-JH648-046.JPG",
            "images/mixed/pexels-photo-257910.jpeg",
            "images/mixed/The_children_of_U.S._Army_personnel_attend_a_Christmas_party_at_Fort_Gordon,_Ga.,_Nov._28,_2008_081128-A-NF756-002.jpg",
            "images/mixed/portrets-1754895_640.jpg",
            "images/mixed/US_Army_51788_'Broncos'_return_home_to_Hawaii.jpg",
            "images/mixed/size0.jpg",
            "images/mixed/101225-N-9914P-015.JPG",
            "images/mixed/091209-F-2580A-005.JPG",
            "images/mixed/091202-F-3241E-001.JPG",
            "images/mixed/Silent_house_party_san_francisco.jpg",
            "images/mixed/Riya_Sen_at_Subhas_Ghai_Christmas_Party_2011.jpg",
            "images/mixed/article_picture_238.jpg",
            "images/mixed/person-1041904_640.jpg",
            "images/mixed/photo-booth-wedding-party-girls-160420.jpeg",
            "images/mixed/141213-F-LM669-185.JPG",
            "images/mixed/subway-2966568_640.jpg",
            "images/mixed/8372563760_face3aff8c_b.jpg",
            "images/mixed/121201-F-JH648-060.JPG",
            "images/mixed/christmas-party.jpg",
            "images/mixed/Group_of_Tajik_Students_in_Karachi_University.jpg",
            "images/mixed/male-1781413_640.png",
            "images/mixed/141219-M-RZ020-001.JPG",
            "images/mixed/people-child-kids-playground-party-joy-kindergarten-652193.jpg"
    };

    private static final String[] SAMPLES_POSITIVE = {
            "images/portraits/s_7457961848_f1ef1c3f0f_b.jpg",
            "images/portraits/s_A_Bigly_Christmas.jpg",
            "images/portraits/s_christmas-business-people-1478194721vrf.jpg",
            "images/portraits/s_santa-958641_960_720.jpg",
            "images/portraits/s_Santa-Hat-Red-Mackerel-Cat-Cute-Funny-Christmas-1898614.jpg"
    };

    private static final String[] SAMPLES_NEGATIVE = {
            "images/portraits/n_2017-04-11-18-49-49-725x485.jpg",
            "images/portraits/n_2017-09-30-05-45-31-1100x589.jpg",
            "images/portraits/n_819px-Hillary_Clinton_official_Secretary_of_State_portrait_crop.jpg",
            "images/portraits/n_girl-1770829_960_720.jpg",
            "images/portraits/n_girl-with-flowers-1374221_640.jpg"
    };

    public static void main(String[] args) throws IOException {
        final Trainer trainer = new Trainer(SAMPLES_POSITIVE, SAMPLES_NEGATIVE);
        trainer.train(5);

        for (String image : ALL_IMAGES) {
            final double score = trainer.check(image);
            if (score > 0.5) {
                System.out.println(String.format("MATCH (%.2f): %s", score, image));
                copyTo("results/positive", image);
            } else {
                System.out.println(String.format("NON-MATCH (%.2f): %s", score, image));
                copyTo("results/negative", image);
            }
        }
    }

    private static void copyTo(String dir, String image) {
        try {
            final File here = new File(App.class.getResource(".").toURI());
            FileUtils.copyFileToDirectory(new File(here, image), new File(here, dir));
        } catch (URISyntaxException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
