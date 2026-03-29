const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status');
const ctx = canvas.getContext('2d');

let model;
let frameCount = 0;
let lastGeminiCheck = 0;
const CHECK_INTERVAL = 60000; // דקה אחת במילישניות
let geminiOffset = { dx: 0, dy: 0 }; // ההזחה שנשמור

// הגדרות המודל - וודא שהן תואמות לייצוא מה-Colab
const IMGSZ = 320;
const CONF_THRESHOLD = 0.45; // סף רגישות (ניתן להוריד אם לא רואים כלום)

async function setupApp() {
    console.log("1. טוען מודל...");
    try {
        // טעינת המודל מהתיקייה
        model = await tf.loadGraphModel('best_web_model/model.json');
        console.log("2. המודל נטען בהצלחה!");
        statusText.innerText = "המודל מוכן, מתחיל בזיהוי...";

        // התחלת הלולאה כשהוידאו מוכן
        if (video.readyState >= 2) {
            startDetection();
        } else {
            video.onloadedmetadata = () => startDetection();
        }
    } catch (e) {
        console.error("❌ שגיאה בטעינה:", e);
        statusText.innerText = "שגיאה בטעינה: " + e.message;
    }
}

async function checkWithGemini(videoElement, yoloBox) {
    console.log("מבצע אימות מול Google AI...");
    
    // 1. חילוץ התמונה מהוידאו בפורמט Base64
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];

    // 2. בניית הפרומפט
    const prompt = `Identify the bounding box of the bird's HEAD in this image. 
    Return ONLY a JSON object in this format: {"ymin": percentage, "xmin": percentage, "ymax": percentage, "xmax": percentage}.
    The percentages should be integers between 0 and 1000 based on the image dimensions.`;

    try {
        // קריאה ל-Gemini (ודא שהגדרת את ה-API Key שלך קודם לכן)
        const response = await model.generateContent([
            prompt,
            { inlineData: { data: base64Image, mimeType: "image/jpeg" } }
        ]);
        
        const text = response.response.text();
        const geminiBox = JSON.parse(text.trim()); // חילוץ ה-JSON
        
        // 3. השוואה וחישוב ההזחה (Offset)
        // נניח ש-yoloBox מגיע בערכים מנורמלים 0-1000
        const yoloCenter = {
            x: (yoloBox.xmin + yoloBox.xmax) / 2,
            y: (yoloBox.ymin + yoloBox.ymax) / 2
        };
        const geminiCenter = {
            x: (geminiBox.xmin + geminiBox.xmax) / 2,
            y: (geminiBox.ymin + geminiBox.ymax) / 2
        };

        // חישוב ההפרש
        geminiOffset.dx = geminiCenter.x - yoloCenter.x;
        geminiOffset.dy = geminiCenter.y - yoloCenter.y;

        console.log(`הזחה מחושבת: X=${geminiOffset.dx.toFixed(2)}, Y=${geminiOffset.dy.toFixed(2)}`);
        
    } catch (error) {
        console.error("שגיאה בפנייה ל-Gemini:", error);
    }
}

function startDetection() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detectFrame();
}

async function detectFrame() {
    if (video.paused || video.ended) return;
    
    frameCount++;

    // מריצים את המודל ומעדכנים את הריבועים רק פעם ב-3 פריימים
    if (frameCount % 3 === 0) {
        
        const detections = tf.tidy(() => {
            // 1. עיבוד מקדים
            let img = tf.browser.fromPixels(video);
            const input = img.resizeBilinear([IMGSZ, IMGSZ])
                             .toFloat()
                             .div(255.0)
                             .expandDims(0);

            // 2. הרצת המודל (כאן נוצר res)
            const res = model.execute(input);

            // 3. עיבוד תוצאות
            const rawBoxes = res[0];
            const rawLogits = res[1];
            const rawScores = tf.sigmoid(rawLogits);

            return {
                boxes: rawBoxes.squeeze().arraySync(),
                scores: rawScores.squeeze().arraySync(),
                classIds: new Array(rawScores.size).fill(0)
            };
        });

        // מציאת התיבה עם הציון הגבוה ביותר
        const bestDetectionIndex = detections.scores.indexOf(Math.max(...detections.scores));
        
        if (bestDetectionIndex !== -1 && detections.scores[bestDetectionIndex] > 0.8) {
            const now = Date.now();
            
            // בדיקה אם עברה דקה מאז הפעם האחרונה
            if (now - lastGeminiCheck > CHECK_INTERVAL) {
                lastGeminiCheck = now;
                
                const bestBox = detections.boxes[bestDetectionIndex]; // נניח שזה אובייקט עם xmin, ymin וכו'
                
                // שליחה ל-Gemini ברקע (בלי לעצור את הוידאו)
                checkWithGemini(video, bestBox);
            }
        }

        // מחילים את ההזחה שנשמרה על התיבות לפני הציור
        const adjustedDetections = {
            ...detections,
            boxes: detections.boxes.map(box => ({
                xmin: box.xmin + geminiOffset.dx,
                xmax: box.xmax + geminiOffset.dx,
                ymin: box.ymin + geminiOffset.dy,
                ymax: box.ymax + geminiOffset.dy
            }))
        };

        // ציור התיבות המוסטות
        drawBoxes(adjustedDetections);
    }
    // 5. בקשה לפריים הבא - תמיד בסוף!
    requestAnimationFrame(detectFrame);
}

function drawBoxes(detections) {
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const { boxes, scores } = detections;
    console.log(boxes);

    for (let i = 0; i < scores.length; i++) {
        if (scores[i] > CONF_THRESHOLD) {
            
            // קבלת הערכים המנורמלים (0-1) כפי שראינו בלוגים
            let [x_center, y_center, width, height] = boxes[i];

            // 1. הגבלת המרכז לתוך גבולות הפריים (0 עד 1)
            // זה מונע מהריבוע "לטפס" או "לרדת" מחוץ לוידאו
            const safeX = Math.max(0, Math.min(x_center, 1));
            const safeY = Math.max(0, Math.min(y_center, 1));

            // 2. חישוב מימדים סופיים בפיקסלים לפי גודל הקנבס האמיתי
            const w = width * canvas.width;
            const h = height * canvas.height;

            // 3. חישוב הפינה השמאלית העליונה (x1, y1)
            // אנחנו מחסירים חצי מהרוחב/גובה מהמרכז המנורמל מוכפל בקנבס
            const x1 = (safeX * canvas.width) - (w / 2);
            const y1 = (safeY * canvas.height) - (h / 2);

            // 4. ציור
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, w, h);

            // הוספת אחוז הביטחון מעל הריבוע
            ctx.fillStyle = "#00FF00";
            ctx.font = "bold 16px Arial";
            ctx.fillText(`${(scores[i] * 100).toFixed(0)}%`, x1, y1 > 20 ? y1 - 5 : 20);
        }
    }
}



setupApp();
