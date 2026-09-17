# Rapporto preliminare sull'archivio raw

## Stato e perimetro

- Archivio verificato in sola lettura: 45 file, 163.719.816 byte, SHA-256 conforme al manifest.
- Provenienza mantenuta tramite il prefisso `Exp...` in ogni record dell'inventario.
- Nessuna pulizia, fusione, finestra, feature, pseudo-label, clustering, classificazione o QML eseguita.
- Output analitici prodotti soltanto in `results/data_quality/preliminary/`.

## Inventario

| Esperimento | SQL | TXT | XLSX | PNG | Totale |
|---|---:|---:|---:|---:|---:|
| Exp20190124 | 2 | 0 | 0 | 0 | 2 |
| Exp20190320 | 3 | 5 | 1 | 4 | 13 |
| Exp20190416 | 8 | 0 | 0 | 0 | 8 |
| Exp20190514 | 3 | 7 | 2 | 0 | 12 |
| Exp20190617 | 4 | 3 | 0 | 3 | 10 |
| Totale | 20 | 15 | 3 | 7 | 45 |

Il ruolo, formato, encoding, delimitatore, righe/record e intervallo di ogni singolo file sono in `file_inventory.csv`.

## Schemi SQL confermati

Tutti i 20 dump sono PostgreSQL UTF-8/ASCII con blocchi `COPY` delimitati da tab.

- `tblPowerLog(ResourceID, timeStamp, ActivePowerL1, Flow, Pressure)`
- `tblOperationLog(ResourceID, timeStamp, Busy, RFIDTagPresent, Done, StationEntryxBG5, ReadyAtStationxBG1, DoneWorkingxBG9, StationExitxBG6, OperationNo, WorkPlanNo, OrderNo, StepNo, CarrierID, iResourceID, OrderPosition, PartNumber)`
- `tblMachineReport(ResourceID, timeStamp, Busy, RFIDTagPresent, Done)`
- `tblSensorsLog(ResourceID, timeStamp, StationEntryxBG5, ReadyAtStationxBG1, DoneWorkingxBG9, StationExitxBG6)`

Nei dump inventariati `tblMachineReport` e `tblSensorsLog` hanno sempre zero righe. I dati effettivi sono in `tblPowerLog` e `tblOperationLog`.

## Intervalli temporali UTC

- Exp20190124: 2019-01-24 09:13:21.693 – 10:02:33.406, in due sessioni separate.
- Exp20190320: 2019-03-20 14:26:46.988 – 16:18:10.719.
- Exp20190416: 2019-04-16 07:23:48.384 – 16:44:47.452, con più sessioni e gap.
- Exp20190514: 2019-05-14 07:28:40.553 – 11:02:50.297, con sessioni 1122 e 1302.
- Exp20190617: 2019-06-17 07:58:16.181 – 12:29:51.424, con un gap tra circa 10:09 e 11:59.

I timestamp SQL sono bigint compatibili con Unix epoch in millisecondi. I probe conservano sia ISO-8601 UTC sia epoch ms; le intestazioni Node-RED visibili sono in ora locale e seguono l'offset stagionale italiano.

## Frequenze osservate

- `tblPowerLog`: eventi irregolari/burst, mediana per ResourceID circa 54–90 ms; p90 tipicamente circa 970–995 ms.
- `tblOperationLog`: eventi di cambio stato, mediana circa 493–533 ms; p90 circa 2,5–5 s.
- Probe `Res 30 line1/line2`: mediana circa 206–218 ms durante le sequenze attive, con lunghi intervalli tra sequenze.
- Non è presente una frequenza unica globale: la pipeline dovrà rispettare la natura event-driven e i gap.

## Workbook Excel

- `DataLogger20190320_4154_1641.xlsx`: un foglio `DataLogger`, A1:F10, 9 record più intestazione, 0 formule, intervallo 14:41:41.335–15:00:48.830 UTC; colonne OPC UA `PrimaryKey, DataType, Value, StatusCode, SourceTimeStamp, ServerTimeStamp`.
- `Elaborazioni_dati_exp20190514_1123_1302.xlsx`: 9 fogli, 1.833 formule memorizzate, calcoli su ingressi/uscite, energia e saving. Contiene almeno due errori formula memorizzati (`#VALUE!`, `#DIV/0!`) e dati stimati/manuali.
- `Expe-summary.xlsx`: un foglio `20190514`, A1:H39, 36 formule, riepilogo di ordini/posizioni e delta temporali.

Limitazione: il loader `@oai/artifact-tool` non è disponibile nella sessione; fogli, dimensioni, valori, formule ed errori sono stati inventariati direttamente dai metadati OOXML, ma non sono stati ricalcolati né validati visualmente tramite il runtime spreadsheet.

## Immagini

Le 7 PNG sono screenshot 1680×1050. Quattro documentano Node-RED, OPC UA e Performance Monitor nel test 20190320; tre mostrano il profilo di potenza della drill station per sequenze di tre, quattro e cinque pezzi nel test 20190617. Sono fonti contestuali/di validazione e non misure tabellari direttamente importabili.

## Qualità e duplicazione

- Nessun gruppo duplicato byte-per-byte rimasto dopo la rimozione richiesta di `dump_20190320_4154 (1).sql`; il payload identico è conservato in `dump_20190320_1641.sql`.
- Cinque coppie `_new` hanno payload identico al corrispondente dump dopo la sola sostituzione del nome schema PostgreSQL: 1718, 1116, 1807, 1302, 1209.
- Snapshot cumulativi: 1718 ricomprende 1641; 1209 ricomprende 1130. Importare tutti i file senza regole duplicherebbe eventi tra file.
- Nessun duplicato di riga esatto rilevato all'interno dei singoli blocchi SQL o dei singoli `query_*`.
- Rilevate inversioni locali dell'ordine temporale in alcuni stream, soprattutto power; occorre ordinare stabilmente per esperimento, ResourceID, timestamp e provenienza senza cancellare record.
- `DoneWorkingxBG9`, `WorkPlanNo` e `StepNo` sono sempre NULL nei dump SQL.
- Nei `query_*`, i vuoti hanno semantica mista: `false` per booleani PLC nelle righe operation e “non applicabile” nelle righe power.
- Range globali osservati: ActivePowerL1 circa 10,93–91,08; Pressure 0–6,08; Flow 0–49,75. Gli estremi di Pressure e Flow richiedono validazione fisica.
- I valori 0 nei campi MES potrebbero essere reset/sentinel e non missing tecnici: non sono stati convertiti.

## Ruolo preliminare delle sorgenti

- **Power log confermato:** `tblPowerLog` nei dump SQL.
- **Operation/PLC/MES log confermato:** `tblOperationLog` nei dump SQL.
- **Sensor log probabile:** probe JSON/Node-RED `Res 30 line1/line2` e workbook DataLogger OPC UA.
- **Esportazioni derivate:** `query_*`, che combinano in forma sparsa righe power e PLC ordinate temporalmente.
- **Documentazione/calcoli derivati:** workbook di elaborazione e summary.
- **Evidenza contestuale:** screenshot PNG.

## Stato della pipeline

Nessun dataset processato o finestra è stato prodotto: numero finestre = 0. Le feature complete e le 6–8 candidate QML non sono state materializzate. Una proposta iniziale, soggetta ad approvazione, è:

- complete: media/min/max/std/energia/slope/delta/durata della potenza; statistiche Pressure/Flow; proporzioni e transizioni PLC; conteggi sensor; riferimenti temporali, esperimento e ResourceID;
- QML candidate: power mean, power std, energia, power delta o slope, pressure mean, flow mean, Busy ratio, numero transizioni StationEntry/StationExit.

Prima di implementare occorre approvare `semantic_decisions_requiring_approval.md`.
