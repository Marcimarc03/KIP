# Begründung des Stage-1 Train/Val/Test-Splits

## Ausgangslage

Der reale Datensatz `object_segmentation_real_v3_1088` umfasst 12 physische
Winkelschleifer ("Tools"), jeweils mehrfach fotografiert. Die Aufteilung erfolgt
**tool-basiert** (ein ganzes Tool liegt vollständig in genau einem Split), weil
Bilder desselben physischen Geräts sonst über Training und Test streuen würden
und das Modell das *Gerät* statt die *Komponente* wiedererkennen könnte
(Data Leakage). Der Split wird auf Bildebene festgelegt, bevor evaluiert wird.

Von den nominell 9 Klassen (`nc=9`, Reihenfolge synchron zum synthetischen
Datensatz) haben nur **6 reale Instanzen**. Drei Klassen
(`anti-vibration_handle`, `intermediate_gearbox`, `wheel_guard`) besitzen in den
Realdaten null Instanzen; sie werden beibehalten, aber als "in Realdaten nicht
vertreten" dokumentiert und in der realen Evaluation nicht bewertet.

## Tool × Bauteil (Instanzenzahl) und Split-Zuordnung

| Tool | bearing\_plate | bevel\_gear\_drive | bevel\_gear\_spindle | gearbox\_housing | motor\_housing | shaft | Split |
|------|---:|---:|---:|---:|---:|---:|:--|
| tool01 | 0 | 28 | 36 | 40 | 0 | 36 | train |
| tool02 | 0 | 17 | 51 | 22 | 35 | 0 | train |
| tool03 | 0 | 13 | 28 | 32 | 28 | 0 | **val** |
| tool04 | 0 | 27 | 0 | 25 | 30 | 0 | train |
| tool05 | 0 | 18 | 18 | 31 | 44 | 0 | train |
| tool08 | 0 | 18 | 35 | 41 | 36 | 0 | train |
| tool09 | 0 | 0 | 13 | 0 | 33 | 0 | train |
| tool10 | 0 | 17 | 26 | 17 | 30 | 0 | train |
| tool13 | 0 | 18 | 0 | 29 | 0 | 0 | train |
| tool97 | 0 | 0 | 4 | 0 | 0 | 0 | train |
| **tool98** | **35** | **12** | **35** | **23** | **23** | **35** | **test** |
| tool99 | 56 | 26 | 64 | 36 | 0 | 56 | train |
| **Σ** | 91 | 194 | 310 | 296 | 259 | 127 | |

Leere Klassen (0 Instanzen in allen Tools): `anti-vibration_handle`,
`intermediate_gearbox`, `wheel_guard`.

## Engpassanalyse

Zwei Klassen sind auf sehr wenige Tools beschränkt und bestimmen daher das Design:

- `bearing_plate`: nur auf **tool98 und tool99** (2 Tools).
- `shaft`: nur auf **tool01, tool98, tool99** (3 Tools).

Alle übrigen realen Klassen liegen auf 8–10 Tools und sind unkritisch.

## Logische Ableitung des Splits

1. **Ziel:** Jede Klasse, auf der trainiert wird, muss auch im Test erscheinen
   (sonst wird sie trainiert, aber nie bewertet) — bei strikter
   Tool-Disjunktheit (kein Tool in zwei Splits).
2. Damit `bearing_plate` *getestet* werden kann, muss eines seiner beiden Tools
   (tool98/tool99) ins Test. Damit es zugleich *trainierbar* bleibt, muss das
   andere im Training bleiben. → genau eines von {tool98, tool99} ins Test.
3. **Welches?** tool98 enthält als einziges Tool **alle 6 realen Klassen**;
   tool99 fehlt `motor_housing`. Test = tool98 deckt somit in einem Zug alle 6
   Klassen ab. → **Test = tool98.**
4. tool99 bleibt im Training → `bearing_plate` trainierbar (56 Instanzen),
   `shaft` trainierbar über tool01 + tool99.
5. **Validierung = tool03** (unverändert aus der gelieferten Aufteilung,
   separates Tool, deckt 4 der 6 Klassen ab).
6. Alle übrigen Tools → Training.

## Ergebnis

| Split | Tools | Bilder | reale Klassen |
|------|------|---:|:--|
| train | tool01,02,04,05,08,09,10,13,97,99 | 803 | alle 6 |
| val | tool03 | 101 | 4 (ohne bearing\_plate, shaft) |
| test | tool98 | 58 | **alle 6** |

Kein Tool liegt in zwei Splits (tool-disjunkt, kein Leakage). Keine Klasse ist
mehr "trainiert, aber nie getestet".

## Warum es vertretbar ist, dass val nicht alle Klassen enthält

Das Validierungsset dient ausschließlich der **Trainingsüberwachung** (Konvergenz,
Auswahl des besten Checkpoints/Early Stopping), nicht der berichteten Endzahl —
diese stammt allein aus dem Test. Dass `bearing_plate` und `shaft` in val fehlen,
beeinflusst daher **nicht die berichteten Ergebnisse**, sondern höchstens minimal
die Checkpoint-Auswahl, die anhand der übrigen (gemeinsam gelernten) Klassen
erfolgt. Zudem ist die Lücke **datenbedingt unvermeidbar**: `bearing_plate`
existiert nur auf zwei physischen Tools, die beide für Training und Test benötigt
werden; unter Tool-Disjunktheit kann eine Klasse mit nur zwei Tools nicht in allen
drei Splits liegen. Wir priorisieren daher bewusst train + test und dokumentieren
die val-Lücke offen.
