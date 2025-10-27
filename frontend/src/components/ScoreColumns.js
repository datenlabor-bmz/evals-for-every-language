import { Column } from 'primereact/column'
import ScoreField from './ScoreField'

const scoreBodyTemplate = (field, options = {}) => {
  const { minScore = 0, maxScore = 1, machineTranslatedMetrics = [] } = options

  return rowData => {
    const score = rowData[field]
    // Prefer per-row flag if present (backend sets `<metric>_is_machine`),
    // otherwise fall back to global list
    const rowFlagKey = `${field}_is_machine`
    const hasRowFlag = Object.prototype.hasOwnProperty.call(rowData, rowFlagKey)
    const isMachineTranslated = hasRowFlag
      ? !!rowData[rowFlagKey]
      : machineTranslatedMetrics.includes(field)
    return ScoreField(score, minScore, maxScore, isMachineTranslated)
  }
}

const ScoreColumns = (machineTranslatedMetrics = []) => [
  <Column
    field='average'
    header='Proficiency'
    headerTooltip='Language Proficiency Score (average of the scores for each task, after min-max normalization)'
    sortable
    body={scoreBodyTemplate('average', { minScore: 0.2, maxScore: 0.5, machineTranslatedMetrics })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='translation_from_bleu'
    header='Translation (from)'
    headerTooltip='Translation performance from a language to all other languages (spBLEU score on a sample of the FLORES+ benchmark)'
    sortable
    body={scoreBodyTemplate('translation_from_bleu', {
      minScore: 0,
      maxScore: 0.5,
      machineTranslatedMetrics
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='translation_to_bleu'
    header='Translation (to)'
    headerTooltip='Translation performance from all other languages to a language (spBLEU score on a sample of the FLORES+ benchmark)'
    sortable
    body={scoreBodyTemplate('translation_to_bleu', {
      minScore: 0,
      maxScore: 0.5,
      machineTranslatedMetrics
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='classification_accuracy'
    header='Classification'
    headerTooltip='Classification performance (accuracy on a sample of the SIB-200 / FLORES+ classification benchmark)'
    sortable
    body={scoreBodyTemplate('classification_accuracy', {
      minScore: 0,
      maxScore: 0.5,
      machineTranslatedMetrics
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  //   <Column
  //     field='language_modeling_chrf'
  //     header='Language Modeling'
  //     sortable
  //     body={scoreBodyTemplate('language_modeling_chrf', {
  //       minScore: 0.8,
  //       maxScore: 1
  //     })}
  //     style={{ minWidth: '5rem', maxWidth: '10rem' }}
  //   />,
  <Column
    field='mmlu_accuracy'
    header='Q&A'
    headerTooltip='Question Answering performance (accuracy on a sample of multilingual versions of the MMLU benchmark)'
    sortable
    body={scoreBodyTemplate('mmlu_accuracy', {
      minScore: 0,
      maxScore: 1,
      machineTranslatedMetrics
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='arc_accuracy'
    header='Advanced Q&A'
    headerTooltip='Advanced Question Answering performance (accuracy on a sample of multilingual versions of the ARC-Easy benchmark)'
    sortable
    body={scoreBodyTemplate('arc_accuracy', {
      minScore: 0,
      maxScore: 1,
      machineTranslatedMetrics
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='mgsm_accuracy'
    header='Math'
    headerTooltip='Math Problem Solving performance (accuracy on a sample of multilingual versions of the GSM8K benchmark)'
    sortable
    body={scoreBodyTemplate('mgsm_accuracy', {
      minScore: 0,
      maxScore: 1,
      machineTranslatedMetrics
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
]

export default ScoreColumns
