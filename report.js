function generateReportPDF() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  // Title
  doc.setFontSize(16);
  doc.text("Social Media Analytics Report", 14, 15);

  // Mock Table Data
  const tableData = [
    [
      "Date Time",
      "Trend Name",
      "Forecast Views",
      "Sentiment",
      "Dynamic (%)",
      "Number of Posts",
      "Viral Posts",
    ],
    ["2026-04-24 10:00", "AI Trend", "12000", "Positive", "78%", "340", "45"],
    ["2026-04-24 10:00", "Crypto Surge", "9500", "Neutral", "65%", "280", "32"],
    ["2026-04-24 10:00", "Sports Buzz", "15000", "Positive", "82%", "410", "60"],
    ["2026-04-24 10:00", "Social Growth", "18000", "Positive", "88%", "500", "75"],
  ];

  // Table
  doc.autoTable({
    head: [tableData[0]],
    body: tableData.slice(1),
    startY: 25,
  });

  // Download PDF
  doc.save("social_media_report.pdf");
}