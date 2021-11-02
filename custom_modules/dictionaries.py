"""custom dictionary library for CIFAR10, CIFAR100 and TinyImageNet dictionaries"""

def get_dicts(used_dataset: int) -> dict:
    """
    helper function which generates and returns a normal and reversed dictionary
    for the CIFAR datasets and the used loss functions.
    :param used_dataset: specifies which dataset dictionary should be returned.
                         0: CIFAR10, 1: CIFAR100, 2: TinyImageNet

    :return: a dictionary with each class and its descriptive label string
    """
    loss_dict = {0:   "BCE_WithLogits",
                 1:   "Wasserstein",
                 2:   "KLDiv",
                 3:   "MinMax"}
    # TinyImageNet
    if used_dataset == 2:
        class_dict = {
            0: "egyptian cat",
            1: "reel",
            2: "volleyball",
            3: "rocking chair",
            4: "lemon",
            5: "bullfrog",
            6: "basketball",
            7: "cliff",
            8: "espresso",
            9: "plunger",
            10: "parking meter",
            11: "german shepherd",
            12: "dining table",
            13: "monarch",
            14: "brown bear",
            15: "school bus",
            16: "pizza",
            17: "guinea pig",
            18: "umbrella",
            19: "organ",
            20: "oboe",
            21: "maypole",
            22: "goldfish",
            23: "potpie",
            24: "hourglass",
            25: "seashore",
            26: "computer keyboard",
            27: "arabian camel",
            28: "ice cream",
            29: "nail",
            30: "space heater",
            31: "cardigan",
            32: "baboon",
            33: "snail",
            34: "coral reef",
            35: "albatross",
            36: "spider web",
            37: "sea cucumber",
            38: "backpack",
            39: "labrador retriever",
            40: "pretzel",
            41: "king penguin",
            42: "sulphur butterfly",
            43: "tarantula",
            44: "lesser panda",
            45: "pop bottle",
            46: "banana",
            47: "sock",
            48: "cockroach",
            49: "projectile",
            50: "beer bottle",
            51: "mantis",
            52: "freight car",
            53: "guacamole",
            54: "remote control",
            55: "european fire salamander",
            56: "lakeside",
            57: "chimpanzee",
            58: "pay-phone",
            59: "fur coat",
            60: "alp",
            61: "lampshade",
            62: "torch",
            63: "abacus",
            64: "moving van",
            65: "barrel",
            66: "tabby",
            67: "goose",
            68: "koala",
            69: "bullet train",
            70: "cd player",
            71: "teapot",
            72: "birdhouse",
            73: "gazelle",
            74: "academic gown",
            75: "tractor",
            76: "ladybug",
            77: "miniskirt",
            78: "golden retriever",
            79: "triumphal arch",
            80: "cannon",
            81: "neck brace",
            82: "sombrero",
            83: "gasmask",
            84: "candle",
            85: "desk",
            86: "frying pan",
            87: "bee",
            88: "dam",
            89: "spiny lobster",
            90: "police van",
            91: "ipod",
            92: "punching bag",
            93: "beacon",
            94: "jellyfish",
            95: "wok",
            96: "potter's wheel",
            97: "sandal",
            98: "pill bottle",
            99: "butcher shop",
            100: "slug",
            101: "hog",
            102: "cougar",
            103: "crane",
            104: "vestment",
            105: "dragonfly",
            106: "cash machine",
            107: "mushroom",
            108: "jinrikisha",
            109: "water tower",
            110: "chest",
            111: "snorkel",
            112: "sunglasses",
            113: "fly",
            114: "limousine",
            115: "black stork",
            116: "dugong",
            117: "sports car",
            118: "water jug",
            119: "suspension bridge",
            120: "ox",
            121: "ice lolly",
            122: "turnstile",
            123: "christmas stocking",
            124: "broom",
            125: "scorpion",
            126: "wooden spoon",
            127: "picket fence",
            128: "rugby ball",
            129: "sewing machine",
            130: "steel arch bridge",
            131: "persian cat",
            132: "refrigerator",
            133: "barn",
            134: "apron",
            135: "yorkshire terrier",
            136: "swimming trunks",
            137: "stopwatch",
            138: "lawn mower",
            139: "thatch",
            140: "fountain",
            141: "black widow",
            142: "bikini",
            143: "plate",
            144: "teddy",
            145: "barbershop",
            146: "confectionery",
            147: "beach wagon",
            148: "scoreboard",
            149: "orange",
            150: "flagpole",
            151: "american lobster",
            152: "trolleybus",
            153: "drumstick",
            154: "dumbbell",
            155: "brass",
            156: "bow tie",
            157: "convertible",
            158: "bighorn",
            159: "orangutan",
            160: "american alligator",
            161: "centipede",
            162: "syringe",
            163: "go-kart",
            164: "brain coral",
            165: "sea slug",
            166: "cliff dwelling",
            167: "mashed potato",
            168: "viaduct",
            169: "military uniform",
            170: "pomegranate",
            171: "chain",
            172: "kimono",
            173: "comic book",
            174: "trilobite",
            175: "bison",
            176: "pole",
            177: "boa constrictor",
            178: "poncho",
            179: "bathtub",
            180: "grasshopper",
            181: "walking stick",
            182: "chihuahua",
            183: "tailed frog",
            184: "lion",
            185: "altar",
            186: "obelisk",
            187: "beaker",
            188: "bell pepper",
            189: "bannister",
            190: "bucket",
            191: "magnetic compass",
            192: "meat loaf",
            193: "gondola",
            194: "standard poodle",
            195: "acorn",
            196: "lifeboat",
            197: "binoculars",
            198: "cauliflower",
            199: "elephant"}

    # CIFAR100
    elif used_dataset == 1:
        class_dict = {0: 'apple',
                      1: 'aquarium_fish',
                      2: 'baby',
                      3: 'bear',
                      4: 'beaver',
                      5: 'bed',
                      6: 'bee',
                      7: 'beetle',
                      8: 'bicycle',
                      9: 'bottle',
                      10: 'bowl',
                      11: 'boy',
                      12: 'bridge',
                      13: 'bus',
                      14: 'butterfly',
                      15: 'camel',
                      16: 'can',
                      17: 'castle',
                      18: 'caterpillar',
                      19: 'cattle',
                      20: 'chair',
                      21: 'chimpanzee',
                      22: 'clock',
                      23: 'cloud',
                      24: 'cockroach',
                      25: 'couch',
                      26: 'crab',
                      27: 'crocodile',
                      28: 'cup',
                      29: 'dinosaur',
                      30: 'dolphin',
                      31: 'elephant',
                      32: 'flatfish',
                      33: 'forest',
                      34: 'fox',
                      35: 'girl',
                      36: 'hamster',
                      37: 'house',
                      38: 'kangaroo',
                      39: 'keyboard',
                      40: 'lamp',
                      41: 'lawn_mower',
                      42: 'leopard',
                      43: 'lion',
                      44: 'lizard',
                      45: 'lobster',
                      46: 'man',
                      47: 'maple_tree',
                      48: 'motorcycle',
                      49: 'mountain',
                      50: 'mouse',
                      51: 'mushroom',
                      52: 'oak_tree',
                      53: 'orange',
                      54: 'orchid',
                      55: 'otter',
                      56: 'palm_tree',
                      57: 'pear',
                      58: 'pickup_truck',
                      59: 'pine_tree',
                      60: 'plain',
                      61: 'plate',
                      62: 'poppy',
                      63: 'porcupine',
                      64: 'possum',
                      65: 'rabbit',
                      66: 'raccoon',
                      67: 'ray',
                      68: 'road',
                      69: 'rocket',
                      70: 'rose',
                      71: 'sea',
                      72: 'seal',
                      73: 'shark',
                      74: 'shrew',
                      75: 'skunk',
                      76: 'skyscraper',
                      77: 'snail',
                      78: 'snake',
                      79: 'spider',
                      80: 'squirrel',
                      81: 'streetcar',
                      82: 'sunflower',
                      83: 'sweet_pepper',
                      84: 'table',
                      85: 'tank',
                      86: 'telephone',
                      87: 'television',
                      88: 'tiger',
                      89: 'tractor',
                      90: 'train',
                      91: 'trout',
                      92: 'tulip',
                      93: 'turtle',
                      94: 'wardrobe',
                      95: 'whale',
                      96: 'willow_tree',
                      97: 'wolf',
                      98: 'woman',
                      99: 'worm'}
    # CIFAR10
    else:
        class_dict = {0:  "airplane",
                      1:  "auto",
                      2:  "bird",
                      3:  "cat",
                      4:  "deer",
                      5:  "dog",
                      6:  "frog",
                      7:  "horse",
                      8:  "ship",
                      9:  "truck"}
    class_dict_rev = {y : x for x, y in class_dict.items()}

    return class_dict, class_dict_rev, loss_dict
