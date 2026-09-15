/* ==========================================================================
   VISIONAI - DUAL-ENGINE 1,000+ REAL-TIME OBJECT DETECTION SYSTEM
   Engine 1: COCO-SSD (80 Spatial Bounding Boxes)
   Engine 2: MobileNet v2 (1,000 Deep Real-World Object Classes)
========================================================================== */

// DOM Elements - Video & Canvas
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const cameraContainer = document.getElementById("cameraContainer");

// DOM Elements - Reticle Scanner
const reticle = document.getElementById("reticle");
const reticleTag = document.getElementById("reticleTag");
const toggleReticleBtn = document.getElementById("toggleReticleBtn");
const primaryIdentified = document.getElementById("primaryIdentified");
const deepPredictions = document.getElementById("deepPredictions");
const deepStatus = document.getElementById("deepStatus");

// DOM Elements - Action Buttons & Status
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");
const clearBtn = document.getElementById("clearBtn");
const cameraStatus = document.getElementById("cameraStatus");
const liveStatus = document.getElementById("liveStatus");
const modelStatus = document.getElementById("modelStatus");
const message = document.getElementById("message");
const objectCount = document.getElementById("objectCount");
const personCount = document.getElementById("personCount");
const confidence = document.getElementById("confidence");
const objectBadge = document.getElementById("objectBadge");
const objectList = document.getElementById("objectList");
const history = document.getElementById("history");

// DOM Elements - Controls & Filters
const aiMode = document.getElementById("aiMode");
const categoryFilter = document.getElementById("categoryFilter");
const confidenceThreshold = document.getElementById("confidenceThreshold");
const confidenceVal = document.getElementById("confidenceVal");
const viewAllObjectsBtn = document.getElementById("viewAllObjectsBtn");
const navDetections = document.getElementById("navDetections");

// DOM Elements - 1,000+ Objects Modal
const objectsModal = document.getElementById("objectsModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const closeModalFooterBtn = document.getElementById("closeModalFooterBtn");
const modalSearchInput = document.getElementById("modalSearchInput");
const modalCategoryPills = document.getElementById("modalCategoryPills");
const allObjectsGrid = document.getElementById("allObjectsGrid");
const modalObjectCount = document.getElementById("modalObjectCount");

// Application State
let cocoModel = null;
let mobilenetModel = null;
let stream = null;
let running = false;
let isClassifying = false;
let isDetecting = false;
let reticleEnabled = true;
let currentMode = "dual"; // "dual", "detector", "classifier"
let minConfidence = 0.30; // 30% default for superior webcam detection
let selectedCategory = "all";
let modalCategoryFilter = "all";
let previousObjects = new Set();
let lastClassifierTime = 0;
let lastTopPrediction = null;

// Offscreen canvas for reticle center crop classification
const cropCanvas = document.createElement("canvas");
cropCanvas.width = 240;
cropCanvas.height = 240;
const cropCtx = cropCanvas.getContext("2d");

/* ==========================================================================
   EXPANDED 1,000+ DETECTABLE OBJECTS COMPREHENSIVE DATABASE
========================================================================== */

const DETECTABLE_CATALOG = [
    // --- OFFICE, TOOLS & EVERYDAY ITEMS (Common items users hold up) ---
    { name: "ballpoint pen", category: "tools", emoji: "🖊️" },
    { name: "fountain pen", category: "tools", emoji: "✒️" },
    { name: "pencil", category: "tools", emoji: "✏️" },
    { name: "pencil sharpener", category: "tools", emoji: "✏️" },
    { name: "eraser / rubber", category: "tools", emoji: "🧼" },
    { name: "ruler", category: "tools", emoji: "📏" },
    { name: "scissors", category: "tools", emoji: "✂️" },
    { name: "stapler", category: "tools", emoji: "📎" },
    { name: "paper clip", category: "tools", emoji: "📎" },
    { name: "envelope", category: "tools", emoji: "✉️" },
    { name: "notebook / binder", category: "tools", emoji: "📓" },
    { name: "book / novel", category: "tools", emoji: "📖" },
    { name: "comic book", category: "tools", emoji: "📚" },
    { name: "screwdriver", category: "tools", emoji: "🪛" },
    { name: "hammer", category: "tools", emoji: "🔨" },
    { name: "wrench / spanner", category: "tools", emoji: "🔧" },
    { name: "pliers", category: "tools", emoji: "🗜️" },
    { name: "power drill", category: "tools", emoji: "🪚" },
    { name: "tape measure", category: "tools", emoji: "📏" },
    { name: "padlock / key lock", category: "tools", emoji: "🔒" },
    { name: "key / brass key", category: "tools", emoji: "🔑" },
    { name: "flashlight / torch", category: "tools", emoji: "🔦" },
    { name: "lighter", category: "tools", emoji: "🔥" },
    { name: "magnifying glass", category: "tools", emoji: "🔍" },
    { name: "microscope", category: "tools", emoji: "🔬" },
    { name: "telescope", category: "tools", emoji: "🔭" },
    { name: "binoculars", category: "tools", emoji: "🔭" },
    { name: "stethoscope", category: "tools", emoji: "🩺" },
    { name: "syringe", category: "tools", emoji: "💉" },
    { name: "band aid", category: "tools", emoji: "🩹" },

    // --- ELECTRONICS & TECH ---
    { name: "cell phone / smartphone", category: "electronics", emoji: "📱" },
    { name: "laptop / notebook pc", category: "electronics", emoji: "💻" },
    { name: "computer mouse", category: "electronics", emoji: "🖱️" },
    { name: "keyboard", category: "electronics", emoji: "⌨️" },
    { name: "monitor / screen", category: "electronics", emoji: "🖥️" },
    { name: "television / tv", category: "electronics", emoji: "📺" },
    { name: "tablet / ipad", category: "electronics", emoji: "📱" },
    { name: "remote control", category: "electronics", emoji: "📲" },
    { name: "headphones / headset", category: "electronics", emoji: "🎧" },
    { name: "earphone / airpods", category: "electronics", emoji: "👂" },
    { name: "digital camera", category: "electronics", emoji: "📷" },
    { name: "webcam", category: "electronics", emoji: "📹" },
    { name: "microphone", category: "electronics", emoji: "🎙️" },
    { name: "speaker / loudspeaker", category: "electronics", emoji: "🔊" },
    { name: "game controller / joystick", category: "electronics", emoji: "🎮" },
    { name: "modem / router", category: "electronics", emoji: "📡" },
    { name: "hard drive / storage", category: "electronics", emoji: "💾" },
    { name: "usb flash drive", category: "electronics", emoji: "💾" },
    { name: "printer / scanner", category: "electronics", emoji: "🖨️" },
    { name: "calculator", category: "electronics", emoji: "🧮" },

    // --- APPAREL, WATCHES & ACCESSORIES ---
    { name: "wristwatch", category: "apparel", emoji: "⌚" },
    { name: "digital watch", category: "apparel", emoji: "⌚" },
    { name: "stopwatch", category: "apparel", emoji: "⏱️" },
    { name: "sunglasses / sunglass", category: "apparel", emoji: "🕶️" },
    { name: "eyeglasses / spectacles", category: "apparel", emoji: "👓" },
    { name: "tie / necktie", category: "apparel", emoji: "👔" },
    { name: "bow tie", category: "apparel", emoji: "🎀" },
    { name: "wallet / billfold", category: "apparel", emoji: "👛" },
    { name: "purse / handbag", category: "apparel", emoji: "👜" },
    { name: "backpack / knapsack", category: "apparel", emoji: "🎒" },
    { name: "suitcase / luggage", category: "apparel", emoji: "🧳" },
    { name: "umbrella", category: "apparel", emoji: "☂️" },
    { name: "running shoe / sneaker", category: "apparel", emoji: "👟" },
    { name: "boot / leather boot", category: "apparel", emoji: "🥾" },
    { name: "sandal / flip flop", category: "apparel", emoji: "🩴" },
    { name: "high heel / stiletto", category: "apparel", emoji: "👠" },
    { name: "sock / stockings", category: "apparel", emoji: "🧦" },
    { name: "glove / mitten", category: "apparel", emoji: "🧤" },
    { name: "baseball cap / hat", category: "apparel", emoji: "🧢" },
    { name: "sombrero / cowboy hat", category: "apparel", emoji: "🤠" },
    { name: "helmet / crash helmet", category: "apparel", emoji: "🪖" },
    { name: "jacket / coat", category: "apparel", emoji: "🧥" },
    { name: "jean / denim pants", category: "apparel", emoji: "👖" },
    { name: "t-shirt / jersey", category: "apparel", emoji: "👕" },

    // --- KITCHEN, DINING & FOOD ---
    { name: "coffee mug", category: "food", emoji: "☕" },
    { name: "teacup", category: "food", emoji: "🍵" },
    { name: "water bottle", category: "food", emoji: "🧴" },
    { name: "wine bottle", category: "food", emoji: "🍾" },
    { name: "wine glass / goblet", category: "food", emoji: "🍷" },
    { name: "beer glass / mug", category: "food", emoji: "🍺" },
    { name: "plate / saucer", category: "food", emoji: "🍽️" },
    { name: "bowl / soup bowl", category: "food", emoji: "🥣" },
    { name: "fork", category: "food", emoji: "🍴" },
    { name: "knife / table knife", category: "food", emoji: "🔪" },
    { name: "spoon / soup spoon", category: "food", emoji: "🥄" },
    { name: "teapot / kettle", category: "food", emoji: "🫖" },
    { name: "frying pan / skillet", category: "food", emoji: "🍳" },
    { name: "pot / saucepan", category: "food", emoji: "🍲" },
    { name: "microwave oven", category: "food", emoji: "📻" },
    { name: "toaster", category: "food", emoji: "🍞" },
    { name: "refrigerator / fridge", category: "food", emoji: "🧊" },
    { name: "blender / mixer", category: "food", emoji: "🍹" },
    { name: "pizza", category: "food", emoji: "🍕" },
    { name: "cheeseburger / burger", category: "food", emoji: "🍔" },
    { name: "hot dog", category: "food", emoji: "🌭" },
    { name: "sandwich", category: "food", emoji: "🥪" },
    { name: "banana", category: "food", emoji: "🍌" },
    { name: "apple", category: "food", emoji: "🍎" },
    { name: "orange", category: "food", emoji: "🍊" },
    { name: "strawberry", category: "food", emoji: "🍓" },
    { name: "broccoli", category: "food", emoji: "🥦" },
    { name: "carrot", category: "food", emoji: "🥕" },
    { name: "donut / doughnut", category: "food", emoji: "🍩" },
    { name: "cake / birthday cake", category: "food", emoji: "🍰" },
    { name: "ice cream / gelato", category: "food", emoji: "🍦" },
    { name: "pop bottle / soda can", category: "food", emoji: "🥤" },

    // --- HOUSEHOLD & FURNITURE ---
    { name: "chair / desk chair", category: "household", emoji: "🪑" },
    { name: "couch / sofa", category: "household", emoji: "🛋️" },
    { name: "dining table", category: "household", emoji: "🍽️" },
    { name: "desk", category: "household", emoji: "🪵" },
    { name: "bed / mattress", category: "household", emoji: "🛏️" },
    { name: "pillow / cushion", category: "household", emoji: "🛏️" },
    { name: "table lamp / light", category: "household", emoji: "💡" },
    { name: "wall clock / clock", category: "household", emoji: "🕐" },
    { name: "analog clock", category: "household", emoji: "🕰️" },
    { name: "vase / flower pot", category: "household", emoji: "🏺" },
    { name: "potted plant", category: "household", emoji: "🪴" },
    { name: "mirror", category: "household", emoji: "🪞" },
    { name: "wardrobe / closet", category: "household", emoji: "🚪" },
    { name: "bookshelf / bookcase", category: "household", emoji: "📚" },
    { name: "trash can / wastebasket", category: "household", emoji: "🗑️" },
    { name: "toilet", category: "household", emoji: "🚽" },
    { name: "sink / washbasin", category: "household", emoji: "🚰" },
    { name: "bathtub", category: "household", emoji: "🛁" },
    { name: "soap dispenser", category: "household", emoji: "🧴" },
    { name: "toothbrush", category: "household", emoji: "🪥" },
    { name: "hair dryer", category: "household", emoji: "💨" },
    { name: "iron / clothing iron", category: "household", emoji: "🥌" },
    { name: "vacuum cleaner", category: "household", emoji: "🧹" },
    { name: "broom", category: "household", emoji: "🧹" },

    // --- MUSICAL INSTRUMENTS ---
    { name: "acoustic guitar", category: "music", emoji: "🎸" },
    { name: "electric guitar", category: "music", emoji: "🎸" },
    { name: "bass guitar", category: "music", emoji: "🎸" },
    { name: "violin / fiddle", category: "music", emoji: "🎻" },
    { name: "cello", category: "music", emoji: "🎻" },
    { name: "piano / grand piano", category: "music", emoji: "🎹" },
    { name: "electronic keyboard / synth", category: "music", emoji: "🎹" },
    { name: "flute", category: "music", emoji: "🪈" },
    { name: "saxophone", category: "music", emoji: "🎷" },
    { name: "trumpet", category: "music", emoji: "🎺" },
    { name: "trombone", category: "music", emoji: "🎺" },
    { name: "drum / snare drum", category: "music", emoji: "🥁" },
    { name: "harmonica", category: "music", emoji: "🪗" },
    { name: "accordion", category: "music", emoji: "🪗" },
    { name: "banjo", category: "music", emoji: "🪕" },
    { name: "harp", category: "music", emoji: "🪕" },
    { name: "maraca", category: "music", emoji: "🪇" },
    { name: "microphone stand", category: "music", emoji: "🎙️" },

    // --- VEHICLES & TRANSPORTATION ---
    { name: "car / passenger car", category: "vehicles", emoji: "🚗" },
    { name: "sports car / race car", category: "vehicles", emoji: "🏎️" },
    { name: "taxi / cab", category: "vehicles", emoji: "🚕" },
    { name: "police car / van", category: "vehicles", emoji: "🚓" },
    { name: "ambulance", category: "vehicles", emoji: "🚑" },
    { name: "fire engine / truck", category: "vehicles", emoji: "🚒" },
    { name: "bus / city bus", category: "vehicles", emoji: "🚌" },
    { name: "school bus", category: "vehicles", emoji: "🚌" },
    { name: "truck / pickup truck", category: "vehicles", emoji: "🚚" },
    { name: "garbage truck", category: "vehicles", emoji: "🚛" },
    { name: "motorcycle / motorbike", category: "vehicles", emoji: "🏍️" },
    { name: "moped / scooter", category: "vehicles", emoji: "🛵" },
    { name: "bicycle / mountain bike", category: "vehicles", emoji: "🚲" },
    { name: "airplane / airliner", category: "vehicles", emoji: "✈️" },
    { name: "helicopter", category: "vehicles", emoji: "🚁" },
    { name: "train / locomotive", category: "vehicles", emoji: "🚆" },
    { name: "boat / speedboat", category: "vehicles", emoji: "🛥️" },
    { name: "canoe / kayak", category: "vehicles", emoji: "🛶" },
    { name: "traffic light / signal", category: "vehicles", emoji: "🚦" },
    { name: "stop sign", category: "vehicles", emoji: "🛑" },
    { name: "fire hydrant", category: "vehicles", emoji: "🧯" },
    { name: "parking meter", category: "vehicles", emoji: "🅿️" },

    // --- ANIMALS & PETS ---
    { name: "dog / puppy", category: "animals", emoji: "🐕" },
    { name: "golden retriever", category: "animals", emoji: "🦮" },
    { name: "german shepherd", category: "animals", emoji: "🐕‍🦺" },
    { name: "bulldog", category: "animals", emoji: "🐶" },
    { name: "beagle", category: "animals", emoji: "🐕" },
    { name: "poodle", category: "animals", emoji: "🐩" },
    { name: "husky", category: "animals", emoji: "🐺" },
    { name: "cat / kitten", category: "animals", emoji: "🐈" },
    { name: "persian cat", category: "animals", emoji: "🐱" },
    { name: "siamese cat", category: "animals", emoji: "🐈" },
    { name: "tabby cat", category: "animals", emoji: "🐱" },
    { name: "bird / robin / sparrow", category: "animals", emoji: "🐦" },
    { name: "parrot / macaw", category: "animals", emoji: "🦜" },
    { name: "eagle / hawk", category: "animals", emoji: "🦅" },
    { name: "owl", category: "animals", emoji: "🦉" },
    { name: "horse", category: "animals", emoji: "🐎" },
    { name: "cow / bull", category: "animals", emoji: "🐄" },
    { name: "sheep / lamb", category: "animals", emoji: "🐑" },
    { name: "pig", category: "animals", emoji: "🐖" },
    { name: "elephant", category: "animals", emoji: "🐘" },
    { name: "bear (grizzly / polar)", category: "animals", emoji: "🐻" },
    { name: "panda", category: "animals", emoji: "🐼" },
    { name: "lion", category: "animals", emoji: "🦁" },
    { name: "tiger", category: "animals", emoji: "🐯" },
    { name: "zebra", category: "animals", emoji: "🦓" },
    { name: "giraffe", category: "animals", emoji: "🦒" },
    { name: "monkey / chimpanzee", category: "animals", emoji: "🐒" },
    { name: "rabbit / bunny", category: "animals", emoji: "🐇" },
    { name: "hamster", category: "animals", emoji: "🐹" },
    { name: "fish / goldfish", category: "animals", emoji: "🐟" },
    { name: "shark", category: "animals", emoji: "🦈" },
    { name: "whale / dolphin", category: "animals", emoji: "🐬" },
    { name: "turtle", category: "animals", emoji: "🐢" },
    { name: "frog", category: "animals", emoji: "🐸" },
    { name: "butterfly", category: "animals", emoji: "🦋" },

    // --- SPORTS, RECREATION & OUTDOORS ---
    { name: "soccer ball", category: "sports", emoji: "⚽" },
    { name: "basketball", category: "sports", emoji: "🏀" },
    { name: "baseball", category: "sports", emoji: "⚾" },
    { name: "tennis ball", category: "sports", emoji: "🎾" },
    { name: "tennis racket", category: "sports", emoji: "🎾" },
    { name: "baseball bat", category: "sports", emoji: "🏏" },
    { name: "baseball glove", category: "sports", emoji: "🥊" },
    { name: "golf ball", category: "sports", emoji: "⛳" },
    { name: "football / rugby ball", category: "sports", emoji: "🏈" },
    { name: "volleyball", category: "sports", emoji: "🏐" },
    { name: "frisbee", category: "sports", emoji: "🥏" },
    { name: "skateboard", category: "sports", emoji: "🛹" },
    { name: "surfboard", category: "sports", emoji: "🏄" },
    { name: "skis / ski poles", category: "sports", emoji: "🎿" },
    { name: "snowboard", category: "sports", emoji: "🏂" },
    { name: "barbell / weight", category: "sports", emoji: "🏋️" },
    { name: "dumbbell", category: "sports", emoji: "🏋️" },
    { name: "boxing glove", category: "sports", emoji: "🥊" },
    { name: "kite", category: "sports", emoji: "🪁" },
    { name: "tent / camping tent", category: "sports", emoji: "⛺" },
    { name: "sleeping bag", category: "sports", emoji: "🏕️" },
    { name: "teddy bear / stuffed toy", category: "sports", emoji: "🧸" },
    { name: "jigsaw puzzle", category: "sports", emoji: "🧩" },

    // --- PEOPLE & MISCELLANEOUS ---
    { name: "person / human", category: "person", emoji: "👤" },
    { name: "face / portrait", category: "person", emoji: "🧑" },
    { name: "child / kid", category: "person", emoji: "🧒" },
    { name: "mask / surgical mask", category: "person", emoji: "😷" }
];

// Helper: map common ImageNet prediction strings to clean labels & emojis
const EMOJI_KEYWORD_MAP = [
    { words: ["pen", "ballpoint", "ballpen", "fountain pen", "quill"], emoji: "🖊️", cat: "tools" },
    { words: ["pencil", "sharpener", "crayon"], emoji: "✏️", cat: "tools" },
    { words: ["ruler", "rule"], emoji: "📏", cat: "tools" },
    { words: ["eraser", "rubber"], emoji: "🧼", cat: "tools" },
    { words: ["scissor", "shears"], emoji: "✂️", cat: "tools" },
    { words: ["watch", "digital watch", "stopwatch", "timepiece"], emoji: "⌚", cat: "apparel" },
    { words: ["sunglass", "sunglasses", "dark glasses", "shades"], emoji: "🕶️", cat: "apparel" },
    { words: ["glass", "spectacles", "eyeglass", "monocle"], emoji: "👓", cat: "apparel" },
    { words: ["phone", "cellular", "cell", "mobile", "smartphone", "iphone", "android"], emoji: "📱", cat: "electronics" },
    { words: ["laptop", "notebook", "computer", "macbook"], emoji: "💻", cat: "electronics" },
    { words: ["mouse", "trackball"], emoji: "🖱️", cat: "electronics" },
    { words: ["keyboard", "keypad"], emoji: "⌨️", cat: "electronics" },
    { words: ["headphone", "headset", "earphone", "airpod"], emoji: "🎧", cat: "electronics" },
    { words: ["camera", "reflex camera", "webcam"], emoji: "📷", cat: "electronics" },
    { words: ["mug", "coffee mug", "cup", "teacup"], emoji: "☕", cat: "food" },
    { words: ["bottle", "water bottle", "beer bottle", "wine bottle", "flask"], emoji: "🍾", cat: "food" },
    { words: ["guitar", "acoustic guitar", "electric guitar"], emoji: "🎸", cat: "music" },
    { words: ["violin", "fiddle", "cello"], emoji: "🎻", cat: "music" },
    { words: ["piano", "grand piano", "upright"], emoji: "🎹", cat: "music" },
    { words: ["shoe", "sneaker", "running shoe", "boot", "sandal", "loafer"], emoji: "👟", cat: "apparel" },
    { words: ["umbrella"], emoji: "☂️", cat: "apparel" },
    { words: ["wallet", "purse", "billfold"], emoji: "👛", cat: "apparel" },
    { words: ["backpack", "knapsack", "rucksack", "bag"], emoji: "🎒", cat: "apparel" },
    { words: ["tie", "necktie", "bow tie"], emoji: "👔", cat: "apparel" },
    { words: ["chair", "armchair", "seat", "stool"], emoji: "🪑", cat: "household" },
    { words: ["couch", "sofa", "settee"], emoji: "🛋️", cat: "household" },
    { words: ["table", "desk", "dining table"], emoji: "🍽️", cat: "household" },
    { words: ["book", "comic", "novel", "binder"], emoji: "📖", cat: "tools" },
    { words: ["clock", "wall clock", "analog clock"], emoji: "🕐", cat: "household" },
    { words: ["dog", "puppy", "hound", "retriever", "terrier", "spaniel"], emoji: "🐕", cat: "animals" },
    { words: ["cat", "kitten", "feline", "tabby", "siamese"], emoji: "🐈", cat: "animals" },
    { words: ["car", "automobile", "sedan", "coupe", "convertible"], emoji: "🚗", cat: "vehicles" },
    { words: ["bicycle", "bike", "cycle"], emoji: "🚲", cat: "vehicles" },
    { words: ["motorcycle", "motorbike", "scooter"], emoji: "🏍️", cat: "vehicles" },
    { words: ["person", "man", "woman", "human", "guy"], emoji: "👤", cat: "person" },
    { words: ["screwdriver", "hammer", "wrench", "pliers", "tool"], emoji: "🔧", cat: "tools" },
    { words: ["key", "padlock", "lock"], emoji: "🔑", cat: "tools" }
];

function getDynamicEmoji(className) {
    const text = (className || "").toLowerCase();
    for (const item of EMOJI_KEYWORD_MAP) {
        for (const w of item.words) {
            if (text.includes(w)) {
                return item.emoji;
            }
        }
    }
    return "📦";
}

function getDynamicCategory(className) {
    const text = (className || "").toLowerCase();
    for (const item of EMOJI_KEYWORD_MAP) {
        for (const w of item.words) {
            if (text.includes(w)) {
                return item.cat;
            }
        }
    }
    return "tools";
}

function formatClassName(rawName) {
    if (!rawName) return "Unknown Object";
    // ImageNet classes often come like "ballpoint, ballpoint pen, ballpen"
    const firstSynonym = rawName.split(",")[0].trim();
    return firstSynonym;
}

/* ==========================================================================
   LOAD BOTH AI MODELS IN PARALLEL (DUAL ENGINE)
========================================================================== */

async function loadAIModels() {
    try {
        modelStatus.textContent = "Loading Dual AI...";
        console.log("Loading COCO-SSD (80 Classes) and MobileNet v2 (1,000 Classes)...");

        const [loadedCoco, loadedMobileNet] = await Promise.all([
            cocoSsd.load(),
            mobilenet.load({ version: 2, alpha: 1.0 })
        ]);

        cocoModel = loadedCoco;
        mobilenetModel = loadedMobileNet;

        modelStatus.textContent = "Ready (1,080+ Objects)";
        console.log("Dual AI Engines successfully loaded! Ready for 1,000+ objects.");
    } catch (error) {
        console.error("Model loading error:", error);
        modelStatus.textContent = "AI Error";
        alert("Failed to load AI models. Please check your internet connection for CDN scripts.");
    }
}

/* ==========================================================================
   START / STOP WEBCAM
========================================================================== */

async function startCamera() {
    if (!cocoModel && !mobilenetModel) {
        alert("AI models are still loading. Please wait a moment.");
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;

        video.onloadedmetadata = () => {
            running = true;
            cameraStatus.textContent = "Online";
            liveStatus.textContent = "● LIVE";
            liveStatus.className = "badge online";
            message.style.display = "none";
            startBtn.disabled = true;
            stopBtn.disabled = false;
            captureBtn.disabled = false;

            // Start the detection and classification loops
            mainAILoop();
        };

    } catch (error) {
        console.error("Camera access denied or error:", error);
        alert(
            "Camera access was denied or is unavailable.\n\n" +
            "Please check browser camera permissions in Chrome settings."
        );
    }
}

function stopCamera() {
    running = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    cameraStatus.textContent = "Offline";
    liveStatus.textContent = "● OFFLINE";
    liveStatus.className = "badge offline";
    message.style.display = "block";

    startBtn.disabled = false;
    stopBtn.disabled = true;
    captureBtn.disabled = true;

    objectCount.textContent = "0";
    personCount.textContent = "0";
    confidence.textContent = "0%";
    objectBadge.textContent = "0";
    primaryIdentified.innerHTML = "<span>🔍</span> Ready to scan";
    reticleTag.textContent = "🎯 Aim at any object";

    deepPredictions.innerHTML = `
        <div class="empty-small">
            <p>Camera offline. Click Start Camera to begin.</p>
        </div>
    `;

    objectList.innerHTML = `
        <div class="empty">
            <div>🔍</div>
            <p>No objects detected</p>
        </div>
    `;
}

/* ==========================================================================
   DUAL-ENGINE MAIN AI LOOP
========================================================================== */

async function mainAILoop() {
    if (!running) return;

    if (video.videoWidth && video.videoHeight) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        // 1. Engine 1: Spatial Bounding Box Detection (COCO-SSD)
        if ((currentMode === "dual" || currentMode === "detector") && cocoModel && !isDetecting) {
            runSpatialDetection();
        }

        // 2. Engine 2: Deep 1,000-Class Object Recognition (MobileNet)
        const now = performance.now();
        if ((currentMode === "dual" || currentMode === "classifier") && mobilenetModel && !isClassifying) {
            // Throttle classification to ~6 times/sec for optimal performance
            if (now - lastClassifierTime > 160) {
                lastClassifierTime = now;
                runDeepClassification();
            }
        }
    }

    if (running) {
        requestAnimationFrame(mainAILoop);
    }
}

/* ==========================================================================
   ENGINE 1: SPATIAL BOUNDING BOX DETECTION (COCO-SSD)
========================================================================== */

async function runSpatialDetection() {
    isDetecting = true;
    try {
        const rawPredictions = await cocoModel.detect(video);

        // Filter by user sensitivity threshold & category
        const filtered = rawPredictions.filter(item => {
            if (item.score < minConfidence) return false;
            if (selectedCategory !== "all") {
                const cat = getDynamicCategory(item.class);
                if (cat !== selectedCategory) return false;
            }
            return true;
        });

        drawBoundingBoxes(filtered);
        updateDashboard(filtered);

    } catch (err) {
        console.error("Spatial detection error:", err);
    } finally {
        isDetecting = false;
    }
}

function drawBoundingBoxes(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const score = Math.round(prediction.score * 100);
        const icon = getDynamicEmoji(prediction.class);
        const label = `${icon} ${prediction.class} ${score}%`;

        // Bounding box border
        ctx.strokeStyle = "#4f46e5";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Label pill background
        ctx.font = "bold 15px Arial, sans-serif";
        const textMetrics = ctx.measureText(label);
        const boxWidth = textMetrics.width + 16;
        const boxHeight = 26;
        const labelY = y >= boxHeight ? y - boxHeight : y;

        ctx.fillStyle = "#4f46e5";
        ctx.fillRect(x, labelY, boxWidth, boxHeight);

        // Label text
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + 8, labelY + 18);
    });
}

function updateDashboard(predictions) {
    objectCount.textContent = predictions.length;
    objectBadge.textContent = predictions.length;

    const people = predictions.filter(item => item.class === "person");
    personCount.textContent = people.length;

    if (predictions.length > 0) {
        const average = predictions.reduce((sum, item) => sum + item.score, 0) / predictions.length;
        confidence.textContent = Math.round(average * 100) + "%";
    } else {
        confidence.textContent = "0%";
    }

    updateObjectList(predictions);
    addNewObjectsToHistory(predictions);
}

function updateObjectList(predictions) {
    if (predictions.length === 0) {
        objectList.innerHTML = `
            <div class="empty">
                <div>🔍</div>
                <p>No bounding boxes detected</p>
            </div>
        `;
        return;
    }

    const counts = {};
    predictions.forEach(prediction => {
        const name = prediction.class;
        if (!counts[name]) {
            counts[name] = { count: 0, confidence: 0 };
        }
        counts[name].count++;
        counts[name].confidence += prediction.score;
    });

    objectList.innerHTML = "";

    Object.entries(counts).forEach(([name, data]) => {
        const average = Math.round((data.confidence / data.count) * 100);
        const item = document.createElement("div");
        item.className = "object-item";

        item.innerHTML = `
            <div>
                <div class="object-name">
                    ${getDynamicEmoji(name)} ${name}
                </div>
                <small>${data.count} detected</small>
            </div>
            <div class="object-confidence">
                ${average}%
            </div>
        `;

        objectList.appendChild(item);
    });
}

/* ==========================================================================
   ENGINE 2: DEEP 1,000-CLASS RECOGNITION (MOBILENET)
========================================================================== */

async function runDeepClassification() {
    isClassifying = true;

    try {
        let inputSource = video;

        // If Center Target Reticle is enabled, crop the center 240x240 region
        if (reticleEnabled && video.videoWidth && video.videoHeight) {
            const cropSize = Math.min(video.videoWidth, video.videoHeight) * 0.45;
            const cropX = (video.videoWidth - cropSize) / 2;
            const cropY = (video.videoHeight - cropSize) / 2;

            cropCtx.drawImage(
                video,
                cropX, cropY, cropSize, cropSize,
                0, 0, cropCanvas.width, cropCanvas.height
            );
            inputSource = cropCanvas;
        }

        // Run deep classification across all 1,000 ImageNet categories
        const results = await mobilenetModel.classify(inputSource, 5);

        if (results && results.length > 0) {
            updateDeepRecognitionHUD(results);
        }

    } catch (err) {
        console.error("Deep classification error:", err);
    } finally {
        isClassifying = false;
    }
}

function updateDeepRecognitionHUD(predictions) {
    if (!predictions || predictions.length === 0) return;

    const top = predictions[0];
    const topScore = Math.round(top.probability * 100);
    const topCleanName = formatClassName(top.className);
    const topEmoji = getDynamicEmoji(top.className);

    lastTopPrediction = {
        name: topCleanName,
        score: topScore,
        emoji: topEmoji
    };

    // Update the center reticle tag
    if (reticleEnabled) {
        reticleTag.textContent = `${topEmoji} ${topCleanName} (${topScore}%)`;
    }

    // Update primary spotlight card
    primaryIdentified.innerHTML = `
        <span>${topEmoji}</span> ${topCleanName}
        <strong style="color: #4f46e5; margin-left: auto; font-size: 14px;">${topScore}%</strong>
    `;

    // Render top predictions with animated confidence bars
    deepPredictions.innerHTML = "";

    predictions.slice(0, 4).forEach((item, index) => {
        const score = Math.round(item.probability * 100);
        const name = formatClassName(item.className);
        const icon = getDynamicEmoji(item.className);

        const row = document.createElement("div");
        row.className = "prediction-row";

        row.innerHTML = `
            <div class="pred-header">
                <span class="pred-name">
                    ${icon} ${name}
                </span>
                <span class="pred-score">
                    ${score}%
                </span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${Math.max(score, 5)}%"></div>
            </div>
        `;

        deepPredictions.appendChild(row);
    });

    // Add high-confidence predictions (> 40%) to history if new
    if (topScore >= 40) {
        addDeepObjectToHistory(topCleanName, topScore, topEmoji);
    }
}

/* ==========================================================================
   HISTORY LOGGING (SPATIAL & DEEP)
========================================================================== */

function addNewObjectsToHistory(predictions) {
    predictions.forEach(prediction => {
        const key = prediction.class;

        if (!previousObjects.has(key)) {
            previousObjects.add(key);
            addHistoryRow(prediction.class, Math.round(prediction.score * 100), getDynamicEmoji(prediction.class));
        }
    });

    setTimeout(() => {
        predictions.forEach(p => previousObjects.delete(p.class));
    }, 6000);
}

function addDeepObjectToHistory(name, score, emoji) {
    const key = `deep_${name.toLowerCase()}`;
    if (!previousObjects.has(key)) {
        previousObjects.add(key);
        addHistoryRow(name, score, emoji);

        setTimeout(() => {
            previousObjects.delete(key);
        }, 8000);
    }
}

function addHistoryRow(name, score, emoji) {
    const empty = history.querySelector(".empty");
    if (empty) {
        empty.parentElement.remove();
    }

    const row = document.createElement("tr");
    const time = new Date().toLocaleTimeString();

    row.innerHTML = `
        <td>${time}</td>
        <td>${emoji} ${name}</td>
        <td><strong>${score}%</strong></td>
    `;

    history.prepend(row);
}

/* ==========================================================================
   SNAPSHOT CAPTURE (FULL COMPOSITE WITH BOUNDING BOXES & RETICLE)
========================================================================== */

captureBtn.addEventListener("click", () => {
    if (!video.videoWidth || !video.videoHeight) {
        alert("Camera feed is not ready for capture.");
        return;
    }

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const captureCtx = captureCanvas.getContext("2d");

    // 1. Draw camera video frame
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    // 2. Overlay bounding boxes canvas
    captureCtx.drawImage(canvas, 0, 0, captureCanvas.width, captureCanvas.height);

    // 3. If reticle is active, draw stylish reticle overlay onto the exported image
    if (reticleEnabled) {
        const cx = captureCanvas.width / 2;
        const cy = captureCanvas.height / 2;
        const half = 110;

        captureCtx.strokeStyle = "#6366f1";
        captureCtx.lineWidth = 4;

        // Top-left bracket
        captureCtx.beginPath();
        captureCtx.moveTo(cx - half, cy - half + 24);
        captureCtx.lineTo(cx - half, cy - half);
        captureCtx.lineTo(cx - half + 24, cy - half);
        captureCtx.stroke();

        // Top-right bracket
        captureCtx.beginPath();
        captureCtx.moveTo(cx + half - 24, cy - half);
        captureCtx.lineTo(cx + half, cy - half);
        captureCtx.lineTo(cx + half, cy - half + 24);
        captureCtx.stroke();

        // Bottom-left bracket
        captureCtx.beginPath();
        captureCtx.moveTo(cx - half, cy + half - 24);
        captureCtx.lineTo(cx - half, cy + half);
        captureCtx.lineTo(cx - half + 24, cy + half);
        captureCtx.stroke();

        // Bottom-right bracket
        captureCtx.beginPath();
        captureCtx.moveTo(cx + half - 24, cy + half);
        captureCtx.lineTo(cx + half, cy + half);
        captureCtx.lineTo(cx + half, cy + half - 24);
        captureCtx.stroke();

        // Center dot
        captureCtx.fillStyle = "#06b6d4";
        captureCtx.beginPath();
        captureCtx.arc(cx, cy, 5, 0, 2 * Math.PI);
        captureCtx.fill();

        // Reticle label
        if (lastTopPrediction) {
            const reticleText = `${lastTopPrediction.emoji} ${lastTopPrediction.name} (${lastTopPrediction.score}%)`;
            captureCtx.font = "bold 15px Arial, sans-serif";
            const tw = captureCtx.measureText(reticleText).width;
            captureCtx.fillStyle = "rgba(15, 23, 42, 0.85)";
            captureCtx.fillRect(cx - tw / 2 - 10, cy + half + 8, tw + 20, 26);
            captureCtx.fillStyle = "#38bdf8";
            captureCtx.fillText(reticleText, cx - tw / 2, cy + half + 26);
        }
    }

    // 4. Timestamp & branding footer
    const bannerHeight = 36;
    captureCtx.fillStyle = "rgba(15, 23, 42, 0.85)";
    captureCtx.fillRect(0, captureCanvas.height - bannerHeight, captureCanvas.width, bannerHeight);

    captureCtx.fillStyle = "#ffffff";
    captureCtx.font = "bold 14px Arial, sans-serif";
    const dateStr = new Date().toLocaleString();
    captureCtx.fillText(`VisionAI 1,000+ Object Detection • ${dateStr}`, 16, captureCanvas.height - 13);

    // 5. Download PNG
    const image = captureCanvas.toDataURL("image/png");
    const link = document.createElement("a");
    const safeTimestamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.download = `visionai-detection-${safeTimestamp}.png`;
    link.href = image;
    link.click();
});

/* ==========================================================================
   UI CONTROLS & EVENT LISTENERS
========================================================================= */

// AI Mode Selector
aiMode.addEventListener("change", (e) => {
    currentMode = e.target.value;
    if (currentMode === "classifier") {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        objectList.innerHTML = `
            <div class="empty">
                <div>🧠</div>
                <p>1,000-Object Scanner active (Bounding boxes disabled)</p>
            </div>
        `;
    }
});

// Toggle Reticle Button
toggleReticleBtn.addEventListener("click", () => {
    reticleEnabled = !reticleEnabled;
    reticle.classList.toggle("hidden", !reticleEnabled);
    toggleReticleBtn.classList.toggle("active", reticleEnabled);
    toggleReticleBtn.textContent = reticleEnabled ? "🎯 Target Scanner: ON" : "🎯 Target Scanner: OFF";
});

// Category Filter Dropdown
categoryFilter.addEventListener("change", (e) => {
    selectedCategory = e.target.value;
});

// Sensitivity Slider
confidenceThreshold.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    minConfidence = val / 100;
    confidenceVal.textContent = `${val}%`;
});

// Clear History Button
clearBtn.addEventListener("click", () => {
    history.innerHTML = `
        <tr>
            <td colspan="3" class="empty">
                No detection history
            </td>
        </tr>
    `;
});

/* ==========================================================================
   ALL 1,000+ DETECTABLE OBJECTS MODAL CATALOG
========================================================================== */

const MODAL_CATEGORIES = [
    { id: "all", label: "All (1,000+)" },
    { id: "tools", label: "🖊️ Office & Tools" },
    { id: "electronics", label: "💻 Tech & Electronics" },
    { id: "apparel", label: "👔 Watches & Apparel" },
    { id: "food", label: "☕ Food & Kitchen" },
    { id: "household", label: "🪑 Home & Furniture" },
    { id: "music", label: "🎸 Musical Instruments" },
    { id: "vehicles", label: "🚗 Vehicles" },
    { id: "animals", label: "🐾 Animals" },
    { id: "sports", label: "⚽ Sports & Outdoor" },
    { id: "person", label: "👤 People" }
];

function initObjectsModal() {
    modalCategoryPills.innerHTML = "";
    MODAL_CATEGORIES.forEach(cat => {
        const pill = document.createElement("button");
        pill.className = `pill-btn ${cat.id === "all" ? "active" : ""}`;
        pill.textContent = cat.label;
        pill.dataset.category = cat.id;

        pill.addEventListener("click", () => {
            modalCategoryPills.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            modalCategoryFilter = cat.id;
            renderModalObjects();
        });

        modalCategoryPills.appendChild(pill);
    });

    renderModalObjects();
}

function renderModalObjects() {
    const query = (modalSearchInput.value || "").toLowerCase().trim();
    allObjectsGrid.innerHTML = "";

    const filtered = DETECTABLE_CATALOG.filter(obj => {
        const matchesCat = modalCategoryFilter === "all" || obj.category === modalCategoryFilter;
        const matchesQuery = !query || obj.name.toLowerCase().includes(query) || obj.category.toLowerCase().includes(query);
        return matchesCat && matchesQuery;
    });

    modalObjectCount.textContent = `Showing ${filtered.length} of ${DETECTABLE_CATALOG.length} catalog items (Model recognizes 1,000+ ImageNet classes)`;

    filtered.forEach(obj => {
        const card = document.createElement("div");
        card.className = "object-card";
        card.title = `Click to filter detection by ${obj.category}`;

        card.innerHTML = `
            <div class="emoji">${obj.emoji}</div>
            <div class="name">${obj.name}</div>
            <div class="cat">${obj.category}</div>
        `;

        card.addEventListener("click", () => {
            if (categoryFilter.querySelector(`option[value="${obj.category}"]`)) {
                categoryFilter.value = obj.category;
                selectedCategory = obj.category;
            }
            closeModal();
        });

        allObjectsGrid.appendChild(card);
    });
}

function openModal() {
    objectsModal.classList.remove("hidden");
    modalSearchInput.value = "";
    modalCategoryFilter = "all";
    modalCategoryPills.querySelectorAll(".pill-btn").forEach(p => {
        p.classList.toggle("active", p.dataset.category === "all");
    });
    renderModalObjects();
    modalSearchInput.focus();
}

function closeModal() {
    objectsModal.classList.add("hidden");
}

modalSearchInput.addEventListener("input", renderModalObjects);
viewAllObjectsBtn.addEventListener("click", openModal);
if (navDetections) {
    navDetections.addEventListener("click", openModal);
}
closeModalBtn.addEventListener("click", closeModal);
closeModalFooterBtn.addEventListener("click", closeModal);

objectsModal.addEventListener("click", (e) => {
    if (e.target === objectsModal) {
        closeModal();
    }
});

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !objectsModal.classList.contains("hidden")) {
        closeModal();
    }
});

/* ==========================================================================
   INITIALIZATION
========================================================================== */

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

initObjectsModal();
loadAIModels();