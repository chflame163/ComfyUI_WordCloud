/**
 * File: debug.js
 * Project: comfy_mtb
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

import * as shared from './comfy_shared.js'
import { log } from './comfy_shared.js'
import { MtbWidgets } from './mtb_widgets.js'

// TODO: respect inputs order...

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}
app.registerExtension({
  name: 'mtb.Debug',
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === 'Debug (mtb)') {
      const onNodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined
        this.addInput(`anything_1`, '*')
        return r
      }

      const onConnectionsChange = nodeType.prototype.onConnectionsChange
      nodeType.prototype.onConnectionsChange = function (
        type,
        index,
        connected,
        link_info
      ) {
        const r = onConnectionsChange
          ? onConnectionsChange.apply(this, arguments)
          : undefined
        // TODO: remove all widgets on disconnect once computed
        shared.dynamic_connection(this, index, connected, 'anything_', '*')

        //- infer type
        if (link_info) {
          const fromNode = this.graph._nodes.find(
            (otherNode) => otherNode.id == link_info.origin_id
          )
          const type = fromNode.outputs[link_info.origin_slot].type
          this.inputs[index].type = type
          // this.inputs[index].label = type.toLowerCase()
        }
        //- restore dynamic input
        if (!connected) {
          this.inputs[index].type = '*'
          this.inputs[index].label = `anything_${index + 1}`
        }
      }

      const onExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments)

        const prefix = 'anything_'

        if (this.widgets) {
          // const pos = this.widgets.findIndex((w) => w.name === "anything_1");
          // if (pos !== -1) {
          for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name !== 'output_to_console') {
              this.widgets[i].onRemoved?.()
            }
          }
          this.widgets.length = 1
        }
        let widgetI = 1

        if (message.text) {
          for (const txt of message.text) {
            const w = this.addCustomWidget(
              MtbWidgets.DEBUG_STRING(`${prefix}_${widgetI}`, escapeHtml(txt))
            )
            w.parent = this
            widgetI++
          }
        }
        if (message.b64_images) {
          for (const img of message.b64_images) {
            const w = this.addCustomWidget(
              MtbWidgets.DEBUG_IMG(`${prefix}_${widgetI}`, img)
            )
            w.parent = this
            widgetI++
          }
          // this.onResize?.(this.size);
          // this.resize?.(this.size)
        }

        this.setSize(this.computeSize())

        this.onRemoved = function () {
          // When removing this node we need to remove the input from the DOM
          for (let y in this.widgets) {
            if (this.widgets[y].canvas) {
              this.widgets[y].canvas.remove()
            }
            shared.cleanupNode(this)
            this.widgets[y].onRemoved?.()
          }
        }
      }
    }
  },
})
