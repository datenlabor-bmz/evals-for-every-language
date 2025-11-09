import { Column } from 'primereact/column'
import ScoreField from './ScoreField'

const scoreBodyTemplate = (field, options = {}) => {
  const {
    minScore = 0,
    maxScore = 1,
    machineTranslatedMetrics = [],
    ciLowerField = null,
    ciUpperField = null
  } = options

  return rowData => {
    const score = rowData[field]
    const rowFlagKey = `${field}_is_machine`
    const hasRowFlag = Object.prototype.hasOwnProperty.call(rowData, rowFlagKey)
    const isMachineTranslated = hasRowFlag
      ? !!rowData[rowFlagKey]
      : machineTranslatedMetrics.includes(field)
    const ciLower = ciLowerField ? rowData[ciLowerField] : null
    const ciUpper = ciUpperField ? rowData[ciUpperField] : null
    return (
      <ScoreField
        score={score}
        minScore={minScore}
        maxScore={maxScore}
        isMachineTranslated={isMachineTranslated}
        ciLower={ciLower}
        ciUpper={ciUpper}
      />
    )
  }
}

const createScoreColumn = (
  field,
  header,
  tooltip,
  minScore,
  maxScore,
  machineTranslatedMetrics
) => (
  <Column
    field={field}
    header={header}
    headerTooltip={tooltip}
    sortable
    body={scoreBodyTemplate(field, {
      minScore,
      maxScore,
      machineTranslatedMetrics,
      ciLowerField: `${field}_ci_lower`,
      ciUpperField: `${field}_ci_upper`
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />
)

const ScoreColumns = (machineTranslatedMetrics = []) => [
  createScoreColumn(
    'average',
    'Proficiency',
    'Language Proficiency Score (average of the scores for each task)',
    0,
    1,
    machineTranslatedMetrics
  ),
  createScoreColumn(
    'translation_from_bleu',
    'Translation (from)',
    'Translation performance from a language to all other languages (spBLEU score on a sample of the FLORES+ benchmark)',
    0,
    1,
    machineTranslatedMetrics
  ),
  createScoreColumn(
    'translation_to_bleu',
    'Translation (to)',
    'Translation performance from all other languages to a language (spBLEU score on a sample of the FLORES+ benchmark)',
    0,
    1,
    machineTranslatedMetrics
  ),
  createScoreColumn(
    'classification_accuracy',
    'Classification',
    'Classification performance (accuracy on a sample of the SIB-200 / FLORES+ classification benchmark)',
    0,
    1,
    machineTranslatedMetrics
  ),
  createScoreColumn(
    'mmlu_accuracy',
    'Q&A',
    'Question Answering performance (accuracy on a sample of multilingual versions of the MMLU benchmark)',
    0,
    1,
    machineTranslatedMetrics
  ),
  createScoreColumn(
    'arc_accuracy',
    'Advanced Q&A',
    'Advanced Question Answering performance (accuracy on a sample of multilingual versions of the ARC-Easy benchmark)',
    0,
    1,
    machineTranslatedMetrics
  ),
  createScoreColumn(
    'mgsm_accuracy',
    'Math',
    'Math Problem Solving performance (accuracy on a sample of multilingual versions of the GSM8K benchmark)',
    0,
    1,
    machineTranslatedMetrics
  )
]

export default ScoreColumns
