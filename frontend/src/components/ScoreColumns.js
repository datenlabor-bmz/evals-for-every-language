import { Column } from 'primereact/column'
import ScoreField from './ScoreField'

const scoreBodyTemplate = (field, options = {}) => {
  const { minScore = 0, maxScore = 1 } = options

  return rowData => {
    const score = rowData[field]
    return ScoreField(score, minScore, maxScore)
  }
}

const ScoreColumns = [
  <Column
    field='average'
    header='Average'
    headerTooltip='Language Proficiency Score (average of all displayed scores)'
    sortable
    body={scoreBodyTemplate('average', { minScore: 0.2, maxScore: 0.5 })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='translation_from_bleu'
    header='Translation (from)'
    headerTooltip='Translation performance from a language to all other languages (spBLEU score on a sample of the FLORES+ benchmark)'
    sortable
    body={scoreBodyTemplate('translation_from_bleu', {
      minScore: 0,
      maxScore: 0.5
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
      maxScore: 0.5
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />,
  <Column
    field='classification_accuracy'
    header='Classification'
    headerTooltip='Classification performance (accuracy on a sample of the FLORES+ benchmark)'
    sortable
    body={scoreBodyTemplate('classification_accuracy', {
      minScore: 0,
      maxScore: 0.5
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
    header='MMLU'
    headerTooltip='Question Answering performance (accuracy on a sample of multilingual versions of the MMLU benchmark)'
    sortable
    body={scoreBodyTemplate('mmlu_accuracy', {
      minScore: 0,
      maxScore: 1
    })}
    style={{ minWidth: '5rem', maxWidth: '10rem' }}
  />
]

export default ScoreColumns
