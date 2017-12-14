package info.magnolia.ai.similarity;

import java.io.IOException;

public class App {

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
        trainer.train(3);
    }
}
