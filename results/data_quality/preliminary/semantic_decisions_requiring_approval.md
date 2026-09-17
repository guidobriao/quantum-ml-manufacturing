# Decisioni semantiche da approvare prima della pipeline

Nessuna delle decisioni seguenti è stata applicata ai dati.

1. **Sorgente primaria.** Proposta: usare i blocchi `COPY` SQL come fonte primaria e i `query_*` come controllo/artefatto derivato. I `query_*` sono esportazioni sparse ordinate temporalmente, non una fusione per timestamp già pronta.
2. **Snapshot duplicati per solo schema.** Le coppie `1718/_new`, `1116/_new`, `1807/_new`, `1302/_new` e `1209/_new` hanno lo stesso payload dopo la sola normalizzazione del nome schema PostgreSQL. Confermare quale membro conservare nella pipeline logica, lasciando comunque intatti i raw.
3. **Snapshot cumulativi.** `1718` include il periodo di `1641`; `1209` include il periodo di `1130`. Confermare se trattare i dump successivi come snapshot cumulativi sostitutivi oppure ricostruire sessioni non sovrapposte.
4. **Duplicato byte-per-byte — decisione già risolta.** `dump_20190320_4154 (1).sql` è stato eliminato su richiesta; il payload identico resta conservato in `dump_20190320_1641.sql`. Non richiede ulteriore approvazione.
5. **Timezone.** Proposta: conservare epoch originale e normalizzare in UTC. Le righe Node-RED mostrano ora locale italiana nel testo ma timestamp ISO con `Z`; usare Europe/Rome soltanto come vista derivata.
6. **ResourceID.** Proposta: conservare `ResourceID` completo e aggiungere un ID stazione normalizzato 10–80. Occorre approvare il mapping tra `eneN_2:CECC-LK`, `mesN_3:*` e la medesima stazione fisica.
7. **Unità.** Il dashboard supporta W per la potenza della drill station. Le unità di `Pressure` e `Flow`, e l'estensione di W a tutte le stazioni, devono essere confermate.
8. **Booleani negli export query.** `BSY/RFD/DNE/INP/WRK/OUT` indicano `true`; il vuoto sembra rappresentare `false` sulle righe PLC ma rappresenta “non applicabile” sulle righe power. Non si può convertire globalmente il vuoto in `false` prima di distinguere il tipo di riga.
9. **WRK e DoneWorking.** Nei `query_*`, `WRK` corrisponde a `ReadyAtStationxBG1`. `DoneWorkingxBG9` è sempre NULL nei dump SQL. Confermare se esiste una sorgente alternativa per DoneWorking.
10. **Zero e NULL nei campi MES.** Confermare se `0` in OrderNo, CarrierID, OperationNo, OrderPosition e PartNumber indica assenza, reset o valore valido.
11. **Probe line1/line2.** Confermare il significato fisico e il fronte attivo dei topic `Res 30 line1` e `Res 30 line2`, nonché il loro rapporto con StationEntry/Exit.
12. **Workbook derivati.** `Elaborazioni_dati...xlsx` ed `Expe-summary.xlsx` contengono formule e calcoli manuali; proposta: usarli come documentazione/validazione, non come raw primario.
13. **Errori Excel.** Sono presenti risultati formula memorizzati `#VALUE!` e `#DIV/0!`; occorre stabilire se derivano da stazioni non disponibili o da errori di calcolo.
14. **Outlier fisici.** Osservati Flow fino a 49,75 contro massimi tipici circa 7–12 e Pressure pari a 0 in una sessione. Confermare plausibilità fisica prima di qualsiasi flag o trattamento.
15. **Regola temporale di fusione.** Proposta: trattare i PLC come cambi di stato con forward-fill limitato all'esperimento/stazione e aggregare i power event nelle finestre; nessun backfill oltre gap o tra esperimenti. Durata massima del mantenimento stato da approvare.
16. **Finestre.** Proposta di confronto: finestre fisse 10 s, 30 s e 60 s, più finestre event-driven tra StationEntry e StationExit. La scelta definitiva deve precedere la produzione del dataset.
