job("Qodana") {
  startOn {
    gitPush {
    }
    codeReviewOpened{}
  }
  container("jetbrains/qodana-python") {
    env["QODANA_TOKEN"] = "{{ project:QODANA_RTM }}"
    shellScript {
      content = "qodana"
    }
  }
}